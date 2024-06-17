# Build and Run
First compile deal.II and libCEED individually. After that, compile the deal.II example:

```bash
mkdir build
cd build
cmake ../ -DDEAL_II_DIR=~/path/to/dealii -DCEED_DIR=~/path/to/libceed
make
```

To run the executable, write:

```
./poisson
```

# Evaluations on Intel(R) Xeon(R) Silver 4116 CPU 
| Cycle No. | No. of DoFs | No. of interations (CEED) | No. of interations (deal.ii) | CEED Operator - Runtime(s) | deal.ii Operator - Runtime(s) |
|-|-|-|-|-|-|
| 0 | 125 | 4 | 4 | 0.00167533 | 0.00275265 |
| 1 | 729 | 11 | 11 | 0.00558121 | 0.0270993 |
| 2 | 4913 | 22 | 22 | 0.0396028 | 0.395944 |
| 3 | 35937 | 46 | 46 | 0.589456 | 6.28851 |
| 4 | 274625 | 94 | 94 | 9.14741 | 100.518 |
| 5 | 2146689 | 100 (Error: solver failed to converge) | 100 (Error: solver failed to converge) | 76.8405 | 852.239 |

# Evaluations on AMD Ryzen Threadripper 2990WX 32-Core CPU
| Cycle No. | No. of DoFs | No. of interations (CEED) | No. of interations (deal.ii) | CEED Operator - Runtime(s) | deal.ii Operator - Runtime(s) |
|-|-|-|-|-|
| 0 | 125 | 4 | 4 | 0.000656872 | 0.00153963 |
| 1 | 729 | 11 | 11 | 0.00187231 | 0.016047 |
| 2 | 4913 | 22 | 22 | 0.0197677 | 0.234204 |
| 3 | 35937 | 46 | 46 | 0.331354 | 3.7008 |
| 4 | 274625 | 94 | 94 | 5.01371 | 62.748 |
| 5 | 2146689 | 100 (Error: solver failed to converge) | 100 (Error: solver failed to converge) | 42.3518 | 503.337 |

# Evaluations on NVIDIA A100-PCIE-40GB GPU
| Cycle No. | No. of DoFs | No. of interations (CEED) | No. of interations (deal.ii) | CEED Operator - Runtime(s) | deal.ii Operator - Runtime(s) |
|-|-|-|-|-|
| 0 | 125 | 4 | 4 | 0.00858444 | 0.00189128 |
| 1 | 729 | 20 | 11 | 0.00374779 | 0.0154138 |
| 2 | 4913 | 46 | 22 | 0.0231386 | 0.255538 |
| 3 | 35937 | 81 | 46 | 0.148143 | 3.75851 |
| 4 | 274625 | 100 (Error: solver failed to converge) | 94 | 1.54341 | 59.5926 |
| 5 | 2146689 | 100 (Error: solver failed to converge) | 100 (Error: solver failed to converge) | 9.98207 | 497.223 |
