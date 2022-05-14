#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 32
#define TW 32
#define MC 32
#define NC 32
#define DOUBLE int 
/*
*********************************************************************
Naive GPU Multiplication of c = a x b
where c = m x n, b = k x n, a = m x k
*********************************************************************
*/

__global__ void naive_gpuMul(int m, int n, int k, int *c,int *a, int *b)
{ 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  if( i < k && j < m) {
    for(int l = 0; l < n; l++) {
      sum += a[i * n + l] * b[l * k + j];
    }
    c[i * k + j] = sum;
  }
}

/*
*********************************************************************
GPU multiplication using shared memory
for squared matrix.
*********************************************************************
*/
__global__ void shared_gpuMul(int N, DOUBLE *C, DOUBLE *A, DOUBLE *B) { 
  //local shared storage
  __shared__ DOUBLE As[MC][TW], Bs[TW][NC];
  DOUBLE Cij  = 0;
  int ty = threadIdx.y;
  int by = blockIdx.y;
  int I = by*NC + ty;
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int J= bx*MC + tx;

  for (int kk=0; kk<((N-1)/TW)+1; kk++) {
    if (((I*N) + ((kk*TW)+tx)) >= N*N) {
      As[ty][tx] = 0;
    } else {
      As[ty][tx] = A[(I*N)+ kk*TW+tx]; 
    }

    if (((((kk*TW)+ty)*N) + J) >= N*N) {
      Bs[ty][tx] = 0; 
    } else {
      Bs[ty][tx] = B[(kk*TW+ty)*N + J];
    }
    __syncthreads();
    
    for (int k=0; k<TW; k++) {
      Cij+=  As[ty][k] * Bs[k][tx];
    }
    __syncthreads();

    if ((I < N) && (J < N)) {
      C[I*N + J] = Cij;
    }
  }
}

/*
*********************************************************************
GPU Transpose of a  and output is c. Size of a is m x n and size of 
c is n x m
*********************************************************************
*/
__global__ void gpu_matrix_transpose(int* a, int* c, unsigned int m, 
                                    unsigned int n) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < n && idy < m) {
    unsigned int pos = idy * n + idx;
    unsigned int trans_pos = idx * m + idy;
    c[trans_pos] = a[pos];
  }
}
/*
*********************************************************************
Naive CPU Multiplication of c = a x b
where c = m x n, b = k x n, a = m x k
*********************************************************************
*/
void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      int tmp = 0.0;
      for (int h = 0; h < n; ++h) {
        tmp += h_a[i * n + h] * h_b[h * k + j];
      }
      h_result[i * k + j] = tmp;
    }
  }
}

/*
*********************************************************************
CPU implementation of matrix transpose
*********************************************************************
*/
void cpu_transpose(int *h_a, int *h_result, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      h_result[j*m + i] = h_a[i*n + j];
    }
  }
}


/*
*********************************************************************
Main function to call the matrix multiplication or transpsoe
Below function is used from the below github link:
https://github.com/lzhengchun/matrix-cuda
*********************************************************************
*/
int main(int argc, char const *argv[]) {
  int isMul;
  srand(3333);
  printf("Please enter 1 for multiplication or use 0 for transpose\n");
  scanf("%d", &isMul);
  if ( isMul == 1) {
    int m, n, k;
    /* Fixed seed for illustration */
    printf("please type in m n and k for matrix multiplication\n");
    scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        h_a[i * n + j] = rand() % 1024;
      }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < k; ++j) {
        h_b[i * k + j] = rand() % 1024;
      }
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
    // Allocate memory space on the device 
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int)*m*k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // Launch kernel 
    if (m == n && n == k) {
      shared_gpuMul<<<dimGrid, dimBlock>>>(n, d_c, d_a, d_b);    
    } else {
      naive_gpuMul<<<dimGrid, dimBlock>>>(m, n, k, d_c, d_a, d_b);    
    } 

    // Transefr results from device to host 
    cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);

    // start the CPU version
    cudaEventRecord(start, 0);

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);

    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < k; ++j) {
        if(h_cc[i*k + j] != h_c[i*k + j]) {
          all_ok = 0;
        }
      }
      //printf("\n");
    }
    // roughly compute speedup
    if(all_ok) {
      printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    } else {
      printf("incorrect results\n");
    }
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    cudaDeviceReset();
  } else {
    int m, n;
    /* Fixed seed for illustration */
    printf("please type in m and n for matrix transpose\n");
    scanf("%d %d", &m, &n);

    // allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
    cudaMallocHost((void **) &h_c, sizeof(int)*n*m);
    cudaMallocHost((void **) &h_cc, sizeof(int)*n*m);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        h_a[i * n + j] = rand() % 1024;
      }
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
    // Allocate memory space on the device 
    int *d_a, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_c, sizeof(int)*n*m);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // Launch kernel 
    gpu_matrix_transpose<<<dimGrid, dimBlock>>>(d_a, d_c, m, n); 

    // Transefr results from device to host 
    cudaMemcpy(h_c, d_c, sizeof(int)*n*m, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    
    printf("Time elapsed on matrix transpose of %dx%d on GPU: %f ms.\n\n", 
    m, n, gpu_elapsed_time_ms);

    // start the CPU version
    cudaEventRecord(start, 0);

    cpu_transpose(h_a, h_cc, m, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix transpose of %dx%d on CPU: %f ms.\n\n",
    m, n, cpu_elapsed_time_ms);

    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if(h_cc[i*m + j] != h_c[i*m + j]) {
          all_ok = 0;
        }
      }
    }
    // roughly compute speedup
    if(all_ok) {
      printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    } else {
      printf("incorrect results\n");
    }
    // free memory
    cudaFree(d_a);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    cudaDeviceReset();
  }
  return 0;
}
