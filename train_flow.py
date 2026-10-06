"""
Train the normalising-flow stage of the PAE on the joint space of AE latent
encodings, redshift, and log-luminosity.

Paths are relative to the repository root; the defaults reproduce the published model.
Requires datasets/sim_sample_AEncoded+z+L.pt. Checkpoints and CSV logs are
written to models/<run_name>/version_<N>/.
"""

####################################################################################################
# Imports & configuration
####################################################################################################

import os
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
import lightning as L
import zuko
from sklearn.model_selection import train_test_split

import models

####################################################################################################
# Argument parsing
####################################################################################################

parser = argparse.ArgumentParser(description='Train normalising flow on AE latent codes.')

# Architecture
parser.add_argument('--flow_type',        type=str,   default='MAF', choices=['MAF', 'NSF', 'CNF', 'NAF'],
                    help='Flow architecture from zuko')
parser.add_argument('--transforms',       type=int,   default=5,
                    help='Number of autoregressive transforms')
parser.add_argument('--hidden_features',  type=str,   default='256,256,256',
                    help='Comma-separated hidden layer widths, e.g. "256,256,256"')

# Data
parser.add_argument('--latent_file',      type=str,
                    default='datasets/sim_sample_AEncoded+z+L.pt',
                    help='Pre-computed latent codes (AE latents + z + logL), shape (N, n_features)')
parser.add_argument('--train_size',       type=int,   default=int(1e6),
                    help='Effective number of samples used for training (rest goes to test/val split)')

# Optimiser
parser.add_argument('--batch_size',       type=int,   default=128)
parser.add_argument('--lr',               type=float, default=1e-3)
parser.add_argument('--optimizer',        type=str,   default='Adam', choices=['Adam', 'AdamW'])
parser.add_argument('--weight_decay',     type=float, default=1e-5,
                    help='L2 regularisation (only applied when --optimizer AdamW)')

# LR scheduler
parser.add_argument('--lr_scheduler',     type=str,   default='reduce',
                    choices=['none', 'reduce', 'cosine'],
                    help='none | reduce (ReduceLROnPlateau) | cosine (CosineAnnealingWarmRestarts)')
parser.add_argument('--plateau_factor',   type=float, default=0.5)
parser.add_argument('--plateau_patience', type=int,   default=10)
parser.add_argument('--cosine_T0',        type=int,   default=10)
parser.add_argument('--cosine_Tmult',     type=int,   default=2)

# Training loop
parser.add_argument('--max_epochs',       type=int,   default=100)
parser.add_argument('--patience',         type=int,   default=25,
                    help='Early-stopping patience in epochs')

args = parser.parse_args()

# ── Derived objects ────────────────────────────────────────────────────────────────────────────

hidden_features = [int(x) for x in args.hidden_features.split(',')]

FLOW_TYPES = {'MAF': zuko.flows.MAF, 'NSF': zuko.flows.NSF, 'CNF': zuko.flows.CNF, 'NAF': zuko.flows.NAF}

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

# Human-readable train-size tag, e.g. 1000000 -> "1000k"
train_size_tag = f"{args.train_size // 1000}k" if args.train_size % 1000 == 0 else str(args.train_size)

run_name = (
    f"Flow_{args.flow_type}"
    f"_t{args.transforms}"
    f"_h{'x'.join(str(w) for w in hidden_features)}"
    f"_{args.optimizer}_lr{args.lr}"
    + (f"_wd{args.weight_decay}"  if optimizer_kwargs               else "")
    + (f"_{args.lr_scheduler}"    if args.lr_scheduler != 'none'    else "")
    + f"_{train_size_tag}_GalSBIUspec1M_restframe"
)

in_slurm = 'SLURM_JOB_ID' in os.environ

print('CUDA available:', torch.cuda.is_available())
print('Run:', run_name)


####################################################################################################
# Data loading
####################################################################################################

# Pre-computed latents: AE encoding + redshift + log-luminosity, shape (N, n_features).
latents = torch.load(args.latent_file)
print('Latents shape:', latents.shape)

####################################################################################################
# Dataset & dataloader construction
####################################################################################################

# Same split formula (and random_state) as train_AE.py so train/test/val sets line up
# index-for-index with the AE's, given the same --train_size and the same underlying N.
idx_train, idx_test = train_test_split(torch.arange(len(latents)), train_size=args.train_size, random_state=151)
idx_test,  idx_val  = train_test_split(idx_test,                   train_size=args.train_size, random_state=151)
print('train:', len(idx_train), '  test:', len(idx_test), '  val:', len(idx_val))

train_loader = DataLoader(latents[idx_train], batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(latents[idx_val],   batch_size=args.batch_size, shuffle=False)



####################################################################################################
# Model definition
####################################################################################################

flow = FLOW_TYPES[args.flow_type](
    features=latents.shape[1],
    context=0,
    transforms=args.transforms,
    hidden_features=hidden_features,
)

flow = models.LFlow(
    model=flow,
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
    enable_progress_bar=not in_slurm,
    callbacks=callbacks,
    logger=logger,
)


####################################################################################################
# Training
####################################################################################################

trainer.fit(
    flow,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    # ckpt_path='path/to/checkpoint.ckpt'  # uncomment to resume from checkpoint
)
