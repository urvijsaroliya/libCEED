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

Optional command-line arguments are shown by adding the command-line argument "--help".
