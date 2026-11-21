# Preregistro — Prior espectral aprendido (H1), subespacio PCA (H5) y consistencia (H2)

**Fecha de registro:** 2026-11-21, antes de implementar los modelos y de entrenar cualquier variante. Este archivo no se modifica después de ver resultados; las desviaciones se agregan al final, con fecha.

## Hipótesis

- **H1 (principal):** un modelo que predice el código latente de un autoencoder entrenado solo con espectros Hyperion, y lo decodifica con ese decodificador congelado, reconstruye mejor el test OLI→Hyperion que el mismo generador prediciendo las 175 bandas directamente (M0).
- **H5 (control de H1):** predecir los coeficientes de una base PCA de Hyperion (decodificador lineal) también mejora sobre M0. Si H1 no supera a H5, el beneficio se atribuye a restringir el espacio de salida y no al AE.
- **H2:** agregar la pérdida de consistencia espectral (fase E-M, M3) mejora sobre la variante sin ella.

## Variantes (mismo generador Hourglass 150/100/75 espacial, mismas mejoras A)

| Brazo | Salida del generador | Decodificador | Pérdida |
| :--- | :--- | :--- | :--- |
| M0 | 175 bandas | — | MSE enmascarado |
| C | 175 bandas | — | MSE + consistencia |
| P | 16 coeficientes | PCA-16 (congelado) | MSE enmascarado |
| A | 16 latentes | AE con cuello de botella, k = 16 (congelado) | MSE enmascarado |
| AC | 16 latentes | AE k = 16 (congelado) | MSE + consistencia |

## Decisiones fijadas de antemano

- **k = 16:** el menor k cuyo error de reconstrucción del subespacio (RMSE 0.011 en reflectancia, 98.6% de la varianza) queda por debajo del 20% del mejor RMSE de test obtenido hasta ahora (~0.06). Se aplica a P y A.
- **AE y PCA** se ajustan solo con espectros Hyperion de las escenas del conjunto de entrenamiento (1–10). Nunca con el test (11–15).
- **Consistencia:** la misma de M3 (D afín 176×7, bandas OLI 2, 4, 5 y 6; coeficiente relativo r = 0.1, λ = 0.319). Sin ajuste nuevo.
- **Datos de entrenamiento final:** las 8 escenas que pasan el control de calidad (1, 2, 3, 4, 7, 8, 9, 10).
- **Número de épocas:** validación cruzada por escena con 4 grupos de 2 escenas ({1, 2}, {3, 4}, {7, 8}, {9, 10}), 1 semilla, hasta 30 épocas con parada temprana (paciencia 6). Para cada brazo, E = mediana de las mejores épocas de los 4 grupos. Los modelos finales se entrenan exactamente E épocas, sin parada temprana.
- **Semillas finales:** 1, 2 y 3 para todos los brazos.
- **Test:** escenas 11–15 completas (2 314 tiles), evaluadas **una sola vez** al final con `scripts/evaluate_models.py`: métricas enmascaradas en reflectancia.

## Análisis

- **Métrica primaria:** RMSE de test (píxeles válidos, reflectancia).
- **Métricas secundarias:** SAM (rad), SSIM, MAE y RMSE por grupo de bandas.
- **Contrastes primarios:** A vs M0 (H1), P vs M0 (H5), A vs P (H1 frente a su control), C vs M0 y AC vs A (H2).
- **Prueba estadística:** Wilcoxon pareado por tile sobre la métrica promediada en las 3 semillas, con corrección de Holm sobre todos los contrastes y métricas (RMSE y SAM).
- **Criterio de confirmación de H1:** A tiene RMSE de test menor que M0 con p_Holm < 0.05, y su SAM no es significativamente peor que el de M0.
- **Análisis secundarios (declarados aquí):** ensamble de las 3 semillas de cada brazo; comparación con los checkpoints de 2020 (solo descriptiva, porque se entrenaron con otros datos y otro protocolo).
- **Se reporta todo**, confirme o no la hipótesis.

## Desviaciones

*(ninguna al momento del registro)*
