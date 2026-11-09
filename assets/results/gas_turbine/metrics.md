# Reproduction metrics

| Mean | K (gap) | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Dispersion ↓ | Iterations | Seconds/iteration | Converged |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Arithmetic | 8 | 0.4272 | 0.7856 | 47275.75 | 27406.41 | 50 | 0.0096 | yes |
| Geometric | 7 | 0.4400 | 0.7289 | 47379.88 | 31390.08 | 37 | 0.0086 | yes |
| Harmonic | 7 | 0.4369 | 0.7296 | 46655.51 | 31820.92 | 45 | 0.0083 | yes |

Quality metrics use two standardized-feature PCA components. Gap uses a seeded 2000-row subset; final fits and dispersion/DB/CH use all 36733 rows. Silhouette and displayed points use the same subset. N/A means a partition has too few/many occupied clusters for that metric. Timing is wall time for all five starts divided by their total iterations, including initialization and final assignment overhead. Iterations reports the selected start only. See metadata.json for unresolved K-search boundaries.
