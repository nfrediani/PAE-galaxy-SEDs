import torch
from torch import nn
from torch import optim
import lightning as L
import zuko


class WeightedMSELoss(nn.Module):
    """
    Mean squared residual in units of the per-pixel uncertainty:
    ``mean(((targets - predictions) / weights)**2)``,
    where ``weights`` are the per-pixel standard deviations (sigma).
    """

    def forward(self, targets, predictions, weights):
        return torch.mean(torch.square((targets - predictions) / weights))


def latent_l2_loss(z):
    """Mean squared L2 norm — keeps latent codes near the origin."""
    return z.pow(2).mean()


def latent_cov_loss(z):
    """
    Off-diagonal covariance penalty — encourages each latent dimension to carry
    independent information. Derived from the VICReg / Barlow Twins covariance term.
    The batch covariance is estimated from the current mini-batch, so a batch size
    of at least ~64 is recommended for a stable estimate.
    """
    B, D = z.shape
    z_c = z - z.mean(dim=0)
    cov = (z_c.T @ z_c) / (B - 1)              # (D, D) empirical covariance
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D


####################################################################################################
# Lightning integration
####################################################################################################

class LModule(L.LightningModule):
    """
    Base Lightning wrapper: configurable optimiser + lr scheduler.

    Subclasses must implement ``_step(batch, log_prefix, on_step) -> loss``
    which handles the forward pass, loss computation, and logging.
    """

    def __init__(self, model,
                 optimizer_cls=optim.Adam, lr=1e-3, optimizer_kwargs=None,
                 lr_scheduler_cls=None, lr_scheduler_kwargs=None):
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, log_prefix, on_step):
        raise NotImplementedError("Subclasses must implement _step().")

    def training_step(self, batch, batch_idx):
        return self._step(batch, log_prefix='train', on_step=True)

    def validation_step(self, batch, batch_idx):
        self._step(batch, log_prefix='val', on_step=False)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), lr=self.hparams.lr,
                                       **self.optimizer_kwargs)
        if self.lr_scheduler_cls is not None:
            scheduler = self.lr_scheduler_cls(optimizer, **self.lr_scheduler_kwargs)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        return optimizer


class LAutoencoder(LModule):
    """
    Lightning wrapper for autoencoders.
    Uses weighted-MSE reconstruction loss with optional latent regularisation.
    """

    def __init__(self, model,
                 loss_fn=None,
                 latent_reg_fn=None,
                 optimizer_cls=optim.Adam, lr=1e-3, optimizer_kwargs=None,
                 lr_scheduler_cls=None, lr_scheduler_kwargs=None):
        super().__init__(model, optimizer_cls, lr, optimizer_kwargs,
                         lr_scheduler_cls, lr_scheduler_kwargs)
        # lr is saved here so configure_optimizers can access self.hparams.lr
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'latent_reg_fn',
                                          'optimizer_cls', 'optimizer_kwargs',
                                          'lr_scheduler_cls', 'lr_scheduler_kwargs'])
        self.loss_fn = loss_fn if loss_fn is not None else WeightedMSELoss()
        self.latent_reg_fn = latent_reg_fn   # callable z -> scalar, or None

    def _step(self, batch, log_prefix, on_step):
        x, weights = batch
        z     = self.model.encode(x)
        x_hat = self.model.decode(z)
        recon = self.loss_fn(x, x_hat, weights)
        if self.latent_reg_fn is not None:
            reg  = self.latent_reg_fn(z)
            loss = recon + reg
            self.log(f'{log_prefix}_recon_loss', recon, on_step=on_step, on_epoch=True)
            self.log(f'{log_prefix}_reg_loss',   reg,   on_step=on_step, on_epoch=True)
        else:
            loss = recon
        self.log(f'{log_prefix}_loss', loss, on_step=on_step, on_epoch=True, prog_bar=True)
        return loss


class LFlow(LModule):
    """
    Lightning wrapper for zuko normalising flows.
    Minimises negative log-likelihood: calls model() to get a distribution, then log_prob(batch).
    """

    def __init__(self, model,
                 optimizer_cls=optim.Adam, lr=1e-3, optimizer_kwargs=None,
                 lr_scheduler_cls=None, lr_scheduler_kwargs=None):
        super().__init__(model, optimizer_cls, lr, optimizer_kwargs,
                         lr_scheduler_cls, lr_scheduler_kwargs)
        self.save_hyperparameters(ignore=['model', 'optimizer_cls', 'optimizer_kwargs',
                                          'lr_scheduler_cls', 'lr_scheduler_kwargs'])

    def _step(self, batch, log_prefix, on_step):
        # zuko flows return a distribution when called; log_prob lives on the distribution
        loss = -self.model().log_prob(batch).mean()
        self.log(f'{log_prefix}_loss', loss, on_step=on_step, on_epoch=True, prog_bar=True)
        return loss


class EpochSummary(L.Callback):
    """Prints one line per epoch — replaces the progress bar in batch jobs."""

    def on_validation_epoch_end(self, trainer, _pl_module):
        m = trainer.logged_metrics
        parts = [f"epoch {trainer.current_epoch:>3d}"]
        for key in ('train_loss_epoch', 'val_loss'):
            if key in m:
                parts.append(f"{key}={m[key]:.5f}")
        print("  ".join(parts), flush=True)


####################################################################################################
# Models
####################################################################################################

class MLP(nn.Module):
    """
    Fully-connected network: a linear layer between consecutive widths in
    ``layers``, with ``activation`` after every layer except the last.
    """

    def __init__(self, layers: list, activation=nn.ReLU):
        super().__init__()
        if len(layers) < 2:
            raise ValueError("layers must contain at least an input and an output width.")
        modules = []
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(activation())
        modules.pop()  # remove last activation
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)


class Autoencoder(nn.Module):
    """
    Autoencoder with customisable encoder and decoder modules.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))


class AsinhLayer(nn.Module):
    """Elementwise ``asinh(x / scale)`` — compresses the dynamic range of the input."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return torch.asinh(x / self.scale)


class SinhLayer(nn.Module):
    """Elementwise ``sinh(x) * scale`` — inverse of :class:`AsinhLayer`."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return torch.sinh(x) * self.scale


####################################################################################################
# Published model configurations
####################################################################################################

N_LATENT = 6  # latent dimensionality of the published autoencoder


def build_paper_autoencoder(n_latent: int = N_LATENT) -> Autoencoder:
    """
    Autoencoder used in the paper (matches the ``train_AE.py`` defaults):
    a symmetric asinh/sinh-wrapped MLP mapping 2000 spectral pixels to
    ``n_latent`` dimensions through hidden widths 1000/500/100.
    """
    return Autoencoder(
        encoder=nn.Sequential(AsinhLayer(scale=1.0),
                              MLP(layers=[2000, 1000, 500, 100, n_latent], activation=nn.ReLU)),
        decoder=nn.Sequential(MLP(layers=[n_latent, 100, 500, 1000, 2000], activation=nn.ReLU),
                              SinhLayer(scale=1.0)),
    )


def build_paper_flow(features: int = N_LATENT + 2) -> zuko.flows.MAF:
    """
    Normalising flow used in the paper (matches the ``train_flow.py``
    defaults): a MAF with 5 transforms of 3x256 hidden features over the
    joint space of (AE latents, z, l*).
    """
    return zuko.flows.MAF(features=features, context=0, transforms=5,
                          hidden_features=[256, 256, 256])
