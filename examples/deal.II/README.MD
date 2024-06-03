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
| Cycle No. | No. of DoFs | No. of interations | CEED Operator - Runtime(s) | deal.ii Operator - Runtime(s) |
|-|-|-|-|-|
| 0 | 125 | 4 | 0.00167533 | 0.00275265 |
| 1 | 729 | 11 | 0.00558121 | 0.0270993 |
| 2 | 4913 | 22 | 0.0396028 | 0.395944 |
| 3 | 35937 | 46 | 0.589456 | 6.28851 |
| 4 | 274625 | 94 | 9.14741 | 100.518 |
| 5 | 2146689 | 100 (Error: solver failed to converge) | 76.8405 | 852.239 |

# TODO: Run on NVIDIA A100 GPU
