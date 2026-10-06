# Data-driven Galaxy Population Prior for Photometric Redshifts

Code accompanying the paper

> N. Frediani, D. Gruen, L. Tortorelli, J. McCullough,
> *Data-driven Galaxy Population Prior for Photometric Redshifts*,
> [arXiv:2609.26594](https://arxiv.org/abs/2609.26594)

We train a **probabilistic autoencoder (PAE)** — an autoencoder compressing
noisy mock galaxy spectra into a low-dimensional latent space, combined with a
normalising flow modelling the joint distribution of latent codes, redshift,
and luminosity — as a purely data-driven model of the galaxy population prior.
Using mock spectra from the GalSBI-SPS galaxy population model, we assess how
well such a model reproduces the colour–redshift relation: galaxies are sorted
into colour-selected tomographic bins with a self-organising map (SOM), and
the mean redshift per bin is compared between the generative model and the
input simulation. Deviations are below Stage-IV weak-lensing requirements,
|Δ⟨z⟩| ≲ 0.0007 (1 + z).

## Repository contents

| File | Purpose |
|---|---|
| `train_AE.py` | Trains the autoencoder on noisy rest-frame galaxy spectra (Sect. 2.2.1 and Appendix A.1 of the paper). |
| `train_flow.py` | Trains the normalising flow on AE latent codes + redshift + log-luminosity (Sect. 2.2.2 and Appendix A.2). |
| `1_SED_model.ipynb` | Evaluation of the trained PAE: spectrum reconstruction, residuals, magnitude accuracy, flow samples and PIT tests. Produces Figs. 2–6 of the paper and the PAE sample used in the next notebook. |
| `2_SED_SOM.ipynb` | Photometry from generated and simulated spectra, SOM assignment, tomographic binning, and the mean-redshift comparison. Produces Figs. 7, 8, C.1, D.1 and Table 1. |
| `models.py` | Model definitions (MLP autoencoder, asinh/sinh layers), PyTorch-Lightning wrappers for AE and flow training, and the published model configurations. |
| `SOM.py` | Minimal self-organising map (loading pre-trained weights and best-matching-unit assignment). |
| `util.py` | Various utility functions. |
| `models/` | Trained autoencoder and flow checkpoints (weights and hyper-parameters only). |
| `datasets/filters/` | Filter transmission curves, one file per band. |
| `datasets/SOM_weights.txt` | Pre-trained C3R2 SOM (McCullough et al. 2024, MNRAS 531, 2582). |
| `figures/` | Figures as they appear in the paper. |

## Data and trained models

The datasets (mock spectra and generated samples) are available on Zenodo: **[DOI: 10.5281/zenodo.22962378](https://zenodo.org/records/22962379)**.
It provides:

```
datasets/
├── filters/                    # included in this repository
├── SOM_weights.txt             # included in this repository
├── sim_sample_sed.pt           # GalSBI-SPS mock rest-frame spectra, shape (2305000, 2000)
├── sim_sample_z.pt             # ... their redshifts
├── sim_sample_AEncoded+z+L.pt  # AE latents + z + l* of the mock spectra, shape (2305000, 8); flow training input
├── PAE_sample_sed.pt           # 10^6 rest-frame spectra sampled from the trained PAE
├── PAE_sample_z.pt             # ... their redshifts
└── PAE_sample_L.pt             # ... their rescaled log-luminosities l* = log10(L* 1e17)
models/
├── AE_n6_enc1000x500x100_ReLU_asinh-sinh_noise0.1_Adam_lr0.001_reduce_1000k_GalSBIUspec1M_restframe/
└── Flow_MAF_t5_h256x256x256_Adam_lr0.001_reduce_1000k_GalSBIUspec1M_restframe/
```

The mock galaxy spectra were generated with the
[GalSBI-SPS] (Tortorelli et al. 2025, A&A 703, A255)
galaxy population model; see the paper for details of the simulation setup.

## Setup

Python 3.12 with the packages in `requirements.txt` (the pinned versions are
the ones used for the paper):

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

A GPU is strongly recommended; the code also runs on CPU, but training and vectorized 
operations are slow. Batching is implemented at critical points to help limit memory usage.

## Reproducing the results

The pipeline runs in this order:

1. **Train the autoencoder** on the noisy mock spectra:
   `python train_AE.py` (defaults reproduce the paper configuration).
   Pre-trained checkpoints are already available.
2. **Encode the training set** into latent codes (AE latents + redshift +
   log-luminosity) — see the latent-encoding cell in `1_SED_model.ipynb`.
   A pre-encoded file is also available. 
3. **Train the normalising flow** on the latent codes:
   `python train_flow.py`.
   Pre-trained checkpoints are already available.
4. **`1_SED_model.ipynb`** — evaluates reconstruction quality and the flow, and
   draws the generated galaxy sample used in the following step.
5. **`2_SED_SOM.ipynb`** — computes photometry, assigns galaxies to the SOM,
   builds tomographic bins, and compares mean redshifts between the generated
   and simulated populations.

With the checkpoints in `models/` and the pre-computed samples from Zenodo,
steps 1–3 can be skipped and the notebooks reproduce the paper figures
directly.

## Citation

If you use this code, please cite the paper (see also `CITATION.cff`):

```bibtex
@article{Frediani2026,
  author  = {Frediani, N. and Gruen, D. and Tortorelli, L. and McCullough, J.},
  title   = {Data-driven Galaxy Population Prior for Photometric Redshifts},
  journal = {arXiv e-prints},
  eprint  = {2609.26594},
  year    = {2026}
}
```

## License

This code is released under the MIT License (see `LICENSE`).
