## Test (escenas 11–15, 2 314 tiles, píxeles válidos, reflectancia)

| Variante | Descripción | RMSE | SAM (rad) | SSIM | MAE | Semillas |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| M0 | MSE (con mejoras A) | 0.0688 ± 0.0044 | 0.3976 ± 0.0152 | 0.358 ± 0.016 | 0.0515 ± 0.0045 | 3 |
| M1 | Perceptual 2020 corregida (AE espacial, Gram bruta) | 23.0446 ± 23.5804 | 1.1471 ± 0.4214 | 0.074 ± 0.127 | 17.1467 ± 17.1786 | 3 |
| M2 | Perceptual mejorada (AE espectral, Gram de correlación, features) | 0.0696 ± 0.0053 | 0.4173 ± 0.0289 | 0.355 ± 0.020 | 0.0523 ± 0.0051 | 3 |
| M3 | M2 + consistencia espectral | 0.0614 ± 0.0033 | 0.4213 ± 0.0043 | 0.409 ± 0.009 | 0.0454 ± 0.0025 | 3 |
| ref. | 2020 UNet + MSE (checkpoint 2020) | 0.0601 | 0.4512 | 0.427 | 0.0405 | 1 |
| ref. | 2020 UNet + Perceptual (checkpoint 2020) | 0.0556 | 0.4549 | 0.415 | 0.0358 | 1 |
| ref. | 2020 CNN + MSE (checkpoint 2020) | 0.0531 | 0.4374 | 0.355 | 0.0364 | 1 |
| ref. | 2020 CNN + Perceptual (checkpoint 2020) | 0.0714 | 0.4354 | 0.395 | 0.0516 | 1 |

## Pruebas pareadas por tile vs M0 (Wilcoxon, corrección de Holm)

| Variante | Métrica | Mediana de la diferencia (variante − M0) | % tiles que mejoran | p (Holm) |
| :--- | :--- | :---: | :---: | :---: |
| M1 | RMSE | +22.83163 | 0.0% | 0.00e+00 |
| M1 | SAM | +0.84378 | 0.3% | 0.00e+00 |
| M2 | RMSE | +0.00043 | 48.0% | 2.81e-17 |
| M2 | SAM | +0.01543 | 32.4% | 3.27e-167 |
| M3 | RMSE | -0.00516 | 65.7% | 3.09e-105 |
| M3 | SAM | +0.01841 | 31.2% | 1.83e-167 |

## RMSE por grupo de bandas (media de semillas)

| Variante | CA | Blue | Green | Red | NIR | SWIR1 | SWIR2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| M0 | 0.0557 | 0.0496 | 0.0461 | 0.0426 | 0.0771 | 0.0670 | 0.0524 |
| M1 | 35.4813 | 26.6998 | 11.6748 | 16.6293 | 19.7465 | 27.1177 | 21.0149 |
| M2 | 0.0664 | 0.0607 | 0.0588 | 0.0494 | 0.0825 | 0.0589 | 0.0462 |
| M3 | 0.0719 | 0.0595 | 0.0513 | 0.0467 | 0.0647 | 0.0461 | 0.0390 |
