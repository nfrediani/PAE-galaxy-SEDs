"""
Train the autoencoder stage of the PAE on noisy, median-normalised rest-frame
galaxy spectra.

Paths are relative to the repository root; the defaults reproduce the published model.
Requires datasets/sim_sample_sed.pt. Checkpoints and CSV logs are written to
models/<run_name>/version_<N>/.
"""

####################################################################################################
# Imports & configuration
####################################################################################################

import os
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from sklearn.model_selection import train_test_split

import models

####################################################################################################
# Argument parsing
####################################################################################################

parser = argparse.ArgumentParser(description='Train MLP autoencoder on rest-frame galaxy SEDs.')

# Architecture
parser.add_argument('--n_latent',        type=int,   default=6)
parser.add_argument('--encoder_hidden',  type=str,   default='1000,500,100',
                    help='Comma-separated hidden layer widths, e.g. "1000,500,100"')
parser.add_argument('--decoder_hidden',  type=str,   default='100,500,1000')
parser.add_argument('--activation',      type=str,   default='ReLU',
                    choices=['ReLU', 'GELU', 'LeakyReLU', 'Sigmoid'])
parser.add_argument('--input_transform', type=str,   default='asinh', choices=['asinh', 'none'])
parser.add_argument('--output_transform',type=str,   default='sinh',  choices=['sinh',  'none'])

# Data
parser.add_argument('--noise_level',     type=float, default=0.1,
                    help='Fractional Gaussian noise: sigma = noise_level * flux')
parser.add_argument('--train_size',      type=int,   default=int(1e6),
                    help='Effective number of samples used for training (rest goes to test/val split)')
  
# Optimiser
parser.add_argument('--batch_size',      type=int,   default=128)
parser.add_argument('--lr',              type=float, default=1e-3)
parser.add_argument('--optimizer',       type=str,   default='Adam', choices=['Adam', 'AdamW'])
parser.add_argument('--weight_decay',    type=float, default=0.0,
                    help='L2 regularisation (only applied when --optimizer AdamW)')

# LR scheduler
parser.add_argument('--lr_scheduler',    type=str,   default='reduce',
                    choices=['none', 'reduce', 'cosine'],
                    help='none | reduce (ReduceLROnPlateau) | cosine (CosineAnnealingWarmRestarts)')
parser.add_argument('--plateau_factor',  type=float, default=0.5)
parser.add_argument('--plateau_patience',type=int,   default=5)
parser.add_argument('--cosine_T0',       type=int,   default=10)
parser.add_argument('--cosine_Tmult',    type=int,   default=2)

# Latent regularisation (weights of 0 disable a term entirely)
parser.add_argument('--latent_l2',       type=float, default=0.0,
                    help='Weight for L2 norm on latent codes')
parser.add_argument('--latent_cov',      type=float, default=0.0,
                    help='Weight for off-diagonal covariance penalty (decorrelation)')

# Training loop
parser.add_argument('--max_epochs',      type=int,   default=100)
parser.add_argument('--patience',        type=int,   default=25,
                    help='Early-stopping patience in epochs')

args = parser.parse_args()

# ── Derived objects ────────────────────────────────────────────────────────────────────────────

ACTIVATIONS     = {'ReLU': nn.ReLU, 'GELU': nn.GELU, 'LeakyReLU': nn.LeakyReLU, 'Sigmoid': nn.Sigmoid}
INPUT_TRANSFORMS = {'asinh': lambda: models.AsinhLayer(scale=1.0), 'none': nn.Identity}
OUTPUT_TRANSFORMS= {'sinh':  lambda: models.SinhLayer(scale=1.0),  'none': nn.Identity}

encoder_hidden = [int(x) for x in args.encoder_hidden.split(',')]
decoder_hidden = [int(x) for x in args.decoder_hidden.split(',')]

optimizer_cls    = {'Adam': optim.Adam, 'AdamW': optim.AdamW}[args.optimizer]
optimizer_kwargs = {'weight_decay': args.weight_decay} if args.optimizer == 'AdamW' else {}

if args.lr_scheduler == 'reduce':
    lr_scheduler_cls    = optim.lr_scheduler.ReduceLROnPlateau
    lr_scheduler_kwargs = {'mode': 'min', 'factor': args.plateau_factor,
                           'patience': args.plateau_patience}
elif args.lr_scheduler == 'cosine':
    lr_scheduler_cls    = optim.lr_scheduler.CosineAnnealingWarmRestarts
    lr_scheduler_kwargs = {'T_0': args.cosine_T0, 'T_mult': args.cosine_Tmult}
else:
    lr_scheduler_cls    = None
    lr_scheduler_kwargs = {}

# Build combined latent regulariser (None when both weights are zero)
_reg_terms = []
if args.latent_l2  > 0: _reg_terms.append(lambda z, w=args.latent_l2:  w * models.latent_l2_loss(z))
if args.latent_cov > 0: _reg_terms.append(lambda z, w=args.latent_cov: w * models.latent_cov_loss(z))
latent_reg_fn = (lambda z: sum(f(z) for f in _reg_terms)) if _reg_terms else None

# Human-readable train-size tag, e.g. 200000 -> "200k"
train_size_tag = f"{args.train_size // 1000}k" if args.train_size % 1000 == 0 else str(args.train_size)

# Run name encodes the full spec — becomes the log directory name
run_name = (
    f"AE_n{args.n_latent}"
    f"_enc{'x'.join(str(w) for w in encoder_hidden)}"
    f"_{args.activation}"
    f"_{args.input_transform}-{args.output_transform}"
    f"_noise{args.noise_level}"
    f"_{args.optimizer}_lr{args.lr}"
    + (f"_wd{args.weight_decay}"    if optimizer_kwargs               else "")
    + (f"_{args.lr_scheduler}"      if args.lr_scheduler != 'none'    else "")
    + (f"_l2{args.latent_l2}"       if args.latent_l2  > 0           else "")
    + (f"_cov{args.latent_cov}"     if args.latent_cov > 0           else "")
    + f"_{train_size_tag}_GalSBIUspec1M_restframe"
)

in_slurm = 'SLURM_JOB_ID' in os.environ

print('CUDA available:', torch.cuda.is_available())
print('Run:', run_name)


####################################################################################################
# Data loading
####################################################################################################


# Rest-frame spectra on a log-spaced wavelength grid: shape (N, 2000)
raw_spectra = torch.load('datasets/sim_sample_sed.pt')


####################################################################################################
# Preprocessing: median normalisation + noise injection
####################################################################################################

# Normalise each spectrum by its median so the AE learns spectral shape, not absolute luminosity
# (in place: raw_spectra is not needed again)
luminosities     = raw_spectra.median(dim=1).values
noiseless_fluxes = raw_spectra.div_(luminosities[:, None])
del raw_spectra

# Fractional noise model: sigma = noise_level * flux, floored at 1e-4 to prevent near-zero
# weights from dominating the weighted-MSE loss
errs = (args.noise_level * noiseless_fluxes).clamp_(min=1e-4)

# Fixed-seed Gaussian noise for reproducibility across runs: fluxes = noiseless_fluxes + errs * N(0, 1)
fluxes = torch.empty_like(noiseless_fluxes).normal_(generator=torch.Generator().manual_seed(151))
fluxes.mul_(errs).add_(noiseless_fluxes)


####################################################################################################
# Dataset & dataloader construction
####################################################################################################

idx_train, idx_test = train_test_split(torch.arange(len(fluxes)), train_size=int(args.train_size), random_state=151)
idx_test,  idx_val  = train_test_split(idx_test,                  train_size=int(args.train_size), random_state=151)
print('train:', len(idx_train), '  test:', len(idx_test), '  val:', len(idx_val))

# Copy out the train/val subsets and free the full-size arrays; only the subsets are needed from here on
train_fluxes, val_fluxes = fluxes[idx_train], fluxes[idx_val]
del fluxes
train_errs, val_errs = errs[idx_train], errs[idx_val]
del errs, noiseless_fluxes

train_dataset = TensorDataset(train_fluxes, train_errs)
val_dataset   = TensorDataset(val_fluxes,   val_errs)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)



####################################################################################################
# Model definition
####################################################################################################

# asinh/sinh wrappers compress the dynamic range of fluxes entering the encoder
# and expand it back before the loss is computed, keeping gradients well-scaled
autoenc = models.Autoencoder(
    encoder=nn.Sequential(
        INPUT_TRANSFORMS[args.input_transform](),
        models.MLP(layers=[2000] + encoder_hidden + [args.n_latent],
                   activation=ACTIVATIONS[args.activation]),
    ),
    decoder=nn.Sequential(
        models.MLP(layers=[args.n_latent] + decoder_hidden + [2000],
                   activation=ACTIVATIONS[args.activation]),
        OUTPUT_TRANSFORMS[args.output_transform](),
    ),
)

autoenc = models.LAutoencoder(
    model=autoenc,
    loss_fn=models.WeightedMSELoss(),
    latent_reg_fn=latent_reg_fn,
    optimizer_cls=optimizer_cls,
    lr=args.lr,
    optimizer_kwargs=optimizer_kwargs,
    lr_scheduler_cls=lr_scheduler_cls,
    lr_scheduler_kwargs=lr_scheduler_kwargs,
)


####################################################################################################
# Logger, callbacks & trainer
####################################################################################################

logger = L.pytorch.loggers.CSVLogger(
    save_dir='models/',
    name=run_name,
    version=None,  # auto-increments so re-submitting the same args creates a new version
)

modelCheckpoint = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=logger.log_dir,
    filename='{epoch}-{val_loss:.5f}',
    monitor='val_loss', mode='min',
    save_top_k=5, save_last=True,
)

earlyStopping = L.pytorch.callbacks.EarlyStopping(
    monitor='val_loss', mode='min',
    patience=args.patience, verbose=False,
)

callbacks = [modelCheckpoint, earlyStopping]
if in_slurm:
    callbacks.append(models.EpochSummary())

trainer = L.Trainer(
    max_epochs=args.max_epochs,
    default_root_dir='models/',
    accelerator='auto',
    devices=1,
    precision='32',
    gradient_clip_val=1.0, gradient_clip_algorithm='norm',
    enable_progress_bar=not in_slurm,
    callbacks=callbacks,
    logger=logger,
)


####################################################################################################
# Training
####################################################################################################

trainer.fit(
    autoenc,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    # ckpt_path='path/to/checkpoint.ckpt'  # uncomment to resume from checkpoint
)
