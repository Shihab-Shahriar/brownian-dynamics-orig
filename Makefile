# You need to have hip and compilers installed.
# PLEASE REPLACE PATHS BELOW WITH YOUR OWN PATHS
# Do this before running the executable:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/ufs18/home-158/khanmd/hippy/clr/build/install/lib


ROOT_PATH = /mnt/ufs18/home-158/khanmd
ROCM_PATH = $(ROOT_PATH)/hippy/clr/build/install
RND_PATH = $(ROOT_PATH)/rnd/include

HOST_COMPILER  = g++
NVCC           = $(ROOT_PATH)/hippy/HIPCC/bin/hipcc -ccbin $(HOST_COMPILER)

#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) 
GENCODE_FLAGS  = -arch=sm_80 
#-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75


SRCS = brownian.cpp
INCS = 
INCLUDE_PATH = -I$(ROCM_PATH)/include -I$(RND_PATH)
LIBS = -L$(ROCM_PATH)/lib

brown: brown.o $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o brown brown.o $(LIBS) $(LDFLAGS)

brown.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDE_PATH) -o brown.o -c brownian.cpp 


out.ppm: brown
	rm -f out.ppm
	./brown > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

profile_basic: brown
	nvprof ./brown > out.ppm

# use nvprof --query-metrics
profile_metrics: brown
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./brown > out.ppm

clean:
	rm -f brown brown.o out.ppm out.jpg out.txt