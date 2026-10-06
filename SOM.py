import numpy as np
import torch
from torch import nn



class SOM(nn.Module):
    """
    Self-organising map for assigning galaxies to a pre-trained codebook
    (loaded via ``load_weights``), working with PyTorch tensors.
    """
    def __init__(
        self,
        map_dims: tuple[int, int],
        feature_dim: int,
        device: torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.map_dims = map_dims
        self.feature_dim = feature_dim
        # Note: self.device is fixed at construction and does not track later .to() moves
        self.device = torch.device(device)
        # Uninitialised until load_weights() fills it
        self.register_buffer(
            "codebook",
            torch.empty(map_dims[0] * map_dims[1], feature_dim, device=self.device),
        )


    def load_weights(
        self,
        path: str,
        columns: tuple[int, ...],
    ) -> None:
        """
        Load SOM weights from a whitespace-separated text file in the Masters
        SOM format (one row per cell); ``columns`` selects the columns holding
        the colour weight vectors.
        """
        data = np.genfromtxt(path)
        weights = torch.as_tensor(
            data[:, columns],
            dtype=self.codebook.dtype,
            device=self.device,
        )
        if weights.shape[1] != self.feature_dim:
            raise ValueError(
                f"Loaded weights feature dimension {weights.shape[1]} "
                f"does not match SOM feature dimension {self.feature_dim}."
            )
        if weights.shape[0] != self.codebook.shape[0]:
            raise ValueError(
                f"Loaded weights contain {weights.shape[0]} codebook vectors "
                f"but SOM expects {self.codebook.shape[0]}."
            )
        self.codebook.copy_(weights)


    def _pairwise_distance(
        self,
        inputs: torch.Tensor,
        codebook: torch.Tensor,
        variance: torch.Tensor | None = None,
        limits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Distance between each input and every codebook vector.

        Without ``limits``: (variance-weighted) squared Euclidean distance.
        With ``limits`` (per-feature flags: 0 = measured, +1 = lower limit on
        the colour, -1 = upper limit, NaN = uninformative): a reduced chi^2
        where measured features contribute normally, limit features contribute
        only when the codebook value violates the bound, and NaN features are
        skipped entirely.

        Returns a (batch, n_units) tensor.
        """
        diff = inputs[:, None, :] - codebook[None, :, :]  # (batch, n_units, feature_dim)

        if limits is None:
            if variance is not None:
                return (diff**2 / variance[:, None, :]).sum(dim=-1)
            return (diff**2).sum(dim=-1)

        lim = limits[:, None, :]  # (batch, 1, feature_dim) — broadcasts over n_units

        good = lim == 0
        blue = lim == 1
        red  = lim == -1

        inconsistent_blue = codebook[None, :, :] < inputs[:, None, :]
        inconsistent_red  = codebook[None, :, :] > inputs[:, None, :]

        chi2 = torch.zeros_like(diff)
        ndof = torch.zeros_like(diff)
        ones = torch.ones_like(diff)

        chi2 = torch.where(good, diff**2 / variance[:, None, :], chi2)
        ndof = torch.where(good, ones, ndof)

        blue_bad = blue & inconsistent_blue
        chi2 = torch.where(blue_bad, diff**2 / 0.75**2, chi2)
        ndof = torch.where(blue_bad, ones, ndof)

        red_bad = red & inconsistent_red
        chi2 = torch.where(red_bad, diff**2 / 0.75**2, chi2)
        ndof = torch.where(red_bad, ones, ndof)

        return chi2.sum(dim=-1) / ndof.sum(dim=-1)


    def find_bmu(
        self,
        inputs: torch.Tensor,
        variance: torch.Tensor | None = None,
        limits: torch.Tensor | None = None,
        chunk_size: int = 10_000,
    ) -> torch.Tensor:
        """
        Find the best matching unit (closest codebook cell) for each input.
        Processing is chunked to limit GPU memory.
        Returns the flat cell indices as a CPU tensor of shape (n_galaxies,).
        """
        if inputs.ndim != 2 or inputs.shape[1] != self.feature_dim:
            raise ValueError("Inputs must have shape (batch_size, feature_dim).")

        num_samples = inputs.shape[0]
        result = torch.empty(num_samples, dtype=torch.long, device=self.device)

        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            inp_c = inputs[start:end].to(self.device, non_blocking=True)
            var_c = variance[start:end].to(self.device, non_blocking=True) if variance is not None else None
            lim_c = limits[start:end].to(self.device, non_blocking=True)   if limits   is not None else None

            distances = self._pairwise_distance(inp_c, self.codebook, var_c, lim_c)
            result[start:end] = torch.argmin(distances, dim=1)

        torch.cuda.empty_cache()
        return result.cpu()


