This is the version that uses AMD's rocRAND library. It works for both AMD and Nvidia GPUs.


## Usage
This uses GNU Make to compile the program. Before running `make`:

1. Make sure you have HIP and rocrand installed in your system and environment variables set as mentioned in `main` branches README.md.
2. Set the `GENCODE_FLAGS` variable to architecture of your GPU.
3. Modify all the paths in the Makefile to point to the your location.
4. run `make`.