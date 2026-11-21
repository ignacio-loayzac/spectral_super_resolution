# Resultado del estudio preregistrado H1 / H5 / H2

**H1 (criterio preregistrado): CONFIRMADA**

## Test (escenas 11–15, 2 314 tiles, píxeles válidos, reflectancia) — media ± d.e. de 3 semillas

| Brazo | Descripción | RMSE | SAM (rad) | SSIM | MAE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| M0 | MSE, 175 bandas | 0.0755 ± 0.0049 | 0.4036 ± 0.0026 | 0.340 ± 0.009 | 0.0580 ± 0.0032 |
| C | MSE + consistencia | 0.0633 ± 0.0051 | 0.3982 ± 0.0111 | 0.398 ± 0.015 | 0.0469 ± 0.0042 |
| P | PCA-16 (decodificador lineal congelado) | 0.0676 ± 0.0036 | 0.4124 ± 0.0088 | 0.382 ± 0.027 | 0.0492 ± 0.0013 |
| A | AE-16 (decodificador no lineal congelado) | 0.0635 ± 0.0023 | 0.4090 ± 0.0025 | 0.398 ± 0.011 | 0.0477 ± 0.0017 |
| AC | AE-16 + consistencia | 0.0596 ± 0.0026 | 0.3953 ± 0.0033 | 0.425 ± 0.013 | 0.0443 ± 0.0021 |

## Contrastes preregistrados (Wilcoxon pareado por tile, Holm sobre 10 pruebas)

| Hipótesis | Contraste | Métrica | Δ media relativa | Mediana Δ por tile | % tiles mejor | p (Holm) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| H1 | A vs M0 | RMSE | -15.9% | -0.00859 | 76.4% | 1.1e-186 |
| H1 | A vs M0 | SAM | +1.3% | -0.00148 | 54.6% | 1.8e-04 |
| H5 | P vs M0 | RMSE | -10.4% | -0.00574 | 70.8% | 2.3e-132 |
| H5 | P vs M0 | SAM | +2.2% | +0.00458 | 35.4% | 1.5e-58 |
| H1 vs control | A vs P | RMSE | -6.1% | -0.00205 | 68.9% | 4.7e-132 |
| H1 vs control | A vs P | SAM | -0.8% | -0.00549 | 61.4% | 7.5e-36 |
| H2 | C vs M0 | RMSE | -16.1% | -0.00857 | 85.5% | 1.8e-286 |
| H2 | C vs M0 | SAM | -1.4% | +0.00150 | 45.9% | 1.1e-01 |
| H2 (latente) | AC vs A | RMSE | -6.1% | -0.00318 | 72.2% | 5.0e-171 |
| H2 (latente) | AC vs A | SAM | -3.4% | -0.00439 | 78.8% | 8.7e-173 |

## RMSE por grupo de bandas (media de semillas)

| Brazo | CA | Blue | Green | Red | NIR | SWIR1 | SWIR2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| M0 | 0.0620 | 0.0530 | 0.0559 | 0.0495 | 0.0796 | 0.0778 | 0.0633 |
| C | 0.0610 | 0.0522 | 0.0517 | 0.0450 | 0.0652 | 0.0542 | 0.0485 |
| P | 0.0632 | 0.0525 | 0.0497 | 0.0425 | 0.0900 | 0.0527 | 0.0406 |
| A | 0.0657 | 0.0541 | 0.0492 | 0.0437 | 0.0757 | 0.0519 | 0.0438 |
| AC | 0.0634 | 0.0526 | 0.0479 | 0.0422 | 0.0651 | 0.0523 | 0.0438 |

## Análisis secundario: ensamble de las 3 semillas

| Ensamble | RMSE | SAM | SSIM |
| :--- | :---: | :---: | :---: |
| M0 ensamble | 0.0729 | 0.3920 | 0.352 |
| C ensamble | 0.0601 | 0.3853 | 0.414 |
| P ensamble | 0.0650 | 0.3801 | 0.399 |
| A ensamble | 0.0615 | 0.3998 | 0.411 |
| AC ensamble | 0.0567 | 0.3785 | 0.440 |

## Referencia descriptiva: checkpoints 2020 (otros datos y protocolo)

| Modelo | RMSE | SAM | SSIM |
| :--- | :---: | :---: | :---: |
| 2020 UNet + MSE | 0.0601 | 0.4512 | 0.427 |
| 2020 UNet + Perceptual | 0.0556 | 0.4549 | 0.415 |
| 2020 CNN + MSE | 0.0531 | 0.4374 | 0.355 |
| 2020 CNN + Perceptual | 0.0714 | 0.4354 | 0.395 |

## Notas de interpretación (escritas DESPUÉS de ver los resultados de test)

1. **H1 se confirma según el criterio preregistrado** (A vs M0: RMSE −15.9%, mejor en el 76% de los tiles, p_Holm ≈ 1e-186; el SAM no empeora de forma significativa en mediana por tile). Ojo: el SAM **medio** de A es 1.3% peor; la mediana por tile es levemente mejor. En la práctica, el prior mejora la magnitud (RMSE, SSIM) y deja la forma espectral (SAM) prácticamente igual.
2. **Robustez por escena (exploratorio, `h1_per_scene_exploratory.md`):** A mejora el RMSE en 4 de 5 escenas de test (−9% a −26%) y empeora en la escena 13 (+12%).
3. **Discrepancia con la validación cruzada:** en CV (escenas de entrenamiento), A era el brazo con peor RMSE medio (0.066 frente a 0.056 de M0). Los resultados de test no se usaron para elegir nada, pero esta discrepancia indica que el efecto depende de las escenas y debe declararse como limitación.
4. **La combinación más fuerte es AC (prior aprendido + consistencia espectral):** RMSE 0.0596 ± 0.0026 (−21% frente a M0), SSIM 0.425, y en ensamble de 3 semillas RMSE 0.0567, SAM 0.379 y SSIM 0.440. La consistencia mejora sobre el prior solo (AC vs A: RMSE −6.1%, SAM −3.4%, ambos con p_Holm < 1e-170).
5. **Frente a los checkpoints de 2020 (comparación solo descriptiva):** AC iguala en RMSE a la UNet+MSE de 2020 (0.0596 frente a 0.0601) con un SAM claramente mejor (0.395 frente a 0.451); el ensamble AC (0.0567) supera a la UNet+MSE de 2020 pero no a la UNet "perceptual" (0.0556) ni a la CNN+MSE (0.0531) de 2020 en RMSE. Esos modelos se entrenaron con otro split, sin máscara y con datos de las escenas 5 y 6.
6. El M0 de este estudio (8 épocas fijas, 8 escenas) es más débil que el M0 de la fase E-M (0.0688); los contrastes son válidos porque todos los brazos comparten exactamente el mismo protocolo.
