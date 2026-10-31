# Studi Komparatif Model Generatif: AE, Sparse AE, VAE, GAN, DCGAN, DDPM

Tugas Take-Home Deep Learning (D6421) Sesi 07. Dataset: Fashion-MNIST (split 80/10/10, seed 42).

## Struktur
- `data.py` : dataset, split 80/10/10, seed tetap
- `models/` : ae, sparse_ae, vae, mlp, frozen_decoder, gan, dcgan, ddpm
- `train_utils.py`, `gan_utils.py`, `ddpm_utils.py` : fungsi training
- `evaluate.py` : MSE / PSNR / SSIM; `benchmark.py` : latensi dan FID
- `figures.py` : Figur 1-5
- `Deeplearning.ipynb` : notebook yang menjalankan seluruh pipeline

Estimasi waktu di GPU T4: AE/VAE ~2.5 menit, GAN ~3 menit, DCGAN ~4 menit, DDPM ~12 menit
(training), FID ~20 menit (DDPM paling lama).
