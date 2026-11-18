# Throughput: Craftax-Classic symbolic, random policy, auto-reset

Environment steps per second (observations included). Best of repeats for JAX; one timed run after warm-up for Mojo.

## flair-node-13, NVIDIA L40S (48 GB)

| num_envs | Mojo GPU | Mojo GPU (naive fused resets) | JAX GPU (optimistic resets) | JAX GPU (auto-reset) |
|---:|---:|---:|---:|---:|
| 1 | 23.8k | – | 421 | 415 |
| 16 | 211.0k | – | 7.8k | 6.0k |
| 256 | 1.70M | – | 100.0k | 80.2k |
| 1,024 | 5.97M | 344.1k | 412.5k | 203.1k |
| 4,096 | 19.11M | – | 753.2k | 242.2k |
| 16,384 | 22.71M | 4.65M | 927.3k | 239.8k |
| 65,536 | 22.99M | – | 1.09M | 238.0k |
| 262,144 | 24.08M | 18.69M | OOM | OOM |
| 1,048,576 | 23.54M | – | – | – |

Peak throughput: Mojo GPU 24.08M, Mojo GPU (naive fused resets) 18.69M, JAX GPU (optimistic resets) 1.09M, JAX GPU (auto-reset) 242.2k.
GPU: peak Mojo / peak JAX = **22.0x**.

## MacBook Pro, Apple M4 Pro (14-core CPU, 20-core GPU)

| num_envs | Mojo GPU | Mojo GPU (naive fused resets) | Mojo CPU | JAX CPU (optimistic resets) | JAX CPU (auto-reset) |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.4k | – | 832.6k | 2.7k | 2.7k |
| 16 | 101.3k | – | 925.1k | 22.4k | 5.6k |
| 256 | 820.6k | – | 2.35M | 51.4k | 10.5k |
| 1,024 | 2.68M | 80.7k | 4.20M | 67.1k | 10.2k |
| 4,096 | 8.46M | – | 6.30M | 70.9k | 8.8k |
| 16,384 | 11.81M | 806.8k | 5.96M | – | – |
| 65,536 | 12.73M | – | – | – | – |
| 262,144 | 14.77M | 3.49M | – | – | – |
| 1,048,576 | 16.12M | – | – | – | – |

Peak throughput: Mojo GPU 16.12M, Mojo GPU (naive fused resets) 3.49M, Mojo CPU 6.30M, JAX CPU (optimistic resets) 70.9k, JAX CPU (auto-reset) 10.5k.
CPU: peak Mojo / peak JAX = **88.9x**.
