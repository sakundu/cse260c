CC=g++
NVCC=nvcc
CXXFLAGS= -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS= -std=c++11 -c 
LIBS= -lcudart -lcublas
LIBDIRS=-L/usr/local/cuda-10.1/lib64
INCDIRS=-I/usr/local/cuda-10.1/include
cuda_submission.o: cuda_submission.cu
	  $(NVCC) $(CUDAFLAGS) cuda_submission.cu 
all: cuda_submission.o 
	  $(CC) -o test cuda_submission.o $(LIBDIRS) $(INCDIRS) $(LIBS) $(CXXFLAGS)
clean:
	    rm -rf test *.o
