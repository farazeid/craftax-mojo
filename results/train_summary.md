# End-to-end PPO: Craftax-Classic-Symbolic

Steps/s is wall time for the whole run after compilation (JAX) / start-up (Mojo). Return, length and score are means over the last 10% of batches; score is the Crafter geometric mean of achievement success rates. ± is the std over seeds.

| machine | config | impl | runs | env steps | steps/s | episode return | episode length | score |
|---|---|---|---:|---:|---:|---:|---:|---:|
| l40s | 4096 | jax | 1 | 49,807,360 | 268k | 12.12 | 318 | 28.42 |
| l40s | 4096 | mojo | 1 | 49,807,360 | 281k | 12.34 | 328 | 29.62 |
| l40s | 512 | jax | 3 | 49,971,200 | 141 ± 1k | 15.62 ± 0.12 | 404 ± 20 | 49.90 ± 2.88 |
| l40s | 512 | mojo | 3 | 49,971,200 | 383 ± 4k | 15.70 ± 0.27 | 408 ± 18 | 51.22 ± 3.86 |
| l40s | 512-fp32 | jax | 1 | 9,994,240 | 116k | 10.76 | 300 | 21.02 |
| l40s | 512-fp32 | mojo | 1 | 9,994,240 | 218k | 10.51 | 294 | 21.55 |

- l40s 4096: Mojo / JAX training throughput = **1.05x**
- l40s 512: Mojo / JAX training throughput = **2.71x**
- l40s 512-fp32: Mojo / JAX training throughput = **1.88x**
