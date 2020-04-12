#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

#define MAXIMUM_VALUE   1000000.0f
#define HANDLE_ERROR( err )  ( HandleError( err, __FILE__, __LINE__ ) )

void HandleError( cudaError_t err, const char *file, int line ) {
  //
  // Handle and report on CUDA errors.
  //
  if ( err != cudaSuccess ) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );

    exit( EXIT_FAILURE );
  }
}

void checkCUDAError( const char *msg, bool exitOnError ) {
  //
  // Check cuda error and print result if appropriate.
  //
  cudaError_t err = cudaGetLastError();

  if( cudaSuccess != err) {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      if (exitOnError) {
        exit(-1);
      }
  }
}

void cleanupCuda( void ) {
  //
  // Clean up CUDA resources.
  //

  //
  // Explicitly cleans up all runtime-related resources associated with the
  // calling host thread.
  //
  HANDLE_ERROR(
         cudaThreadExit()
         );
}

__device__ double device_pow( double x, double y ) {
  //
  // Calculate x^y on the GPU.
  //
  return pow( x, y );
}

//
// PLACE GPU KERNELS HERE - BEGIN
//

__device__ double atomicAddInternal(double* address, double val){
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicMin(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_ull, assumed, __double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicMax(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_ull, assumed, __double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void reduce(double *array, double *sum, double *min, double *max, int n){
  __shared__ double sdata[256];
  // each thread loads one element from global to shared memory
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n){
    sdata[threadIdx.x] = array[i];
  }

  __syncthreads();

  // calculate min and max
  if(blockIdx.x * blockDim.x + threadIdx.x < n){
    atomicMin(min, sdata[threadIdx.x]);
    atomicMax(max, sdata[threadIdx.x]);
  }

  // do reduction in shared memory
  if(blockIdx.x * blockDim.x + threadIdx.x < n){
    for(int s=1; s<blockDim.x; s *=2){
      int index = 2 * s * threadIdx.x;
      if(index < blockDim.x){
        sdata[index] += sdata[index + s];
      }
      __syncthreads();
    }
  }

  // write result for this block to global memory
  if(threadIdx.x == 0){
    atomicAddInternal(sum, sdata[0]);
  }
}

__global__ void reducesquarediff(double *array, double *sum, double *mean, int n){
  __shared__ double sdata[256];
  // each thread loads one element from global, calculates the squared differences, and then store to shared memory
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n){
    sdata[threadIdx.x] = device_pow(array[i] - mean[0], 2);
  }

  __syncthreads();

  // do reduction in shared memory
  if(blockIdx.x * blockDim.x + threadIdx.x < n){
    for(int s=1; s<blockDim.x; s *=2){
      int index = 2 * s * threadIdx.x;
      if(index < blockDim.x){
        sdata[index] += sdata[index + s];
      }
      __syncthreads();
    }
  }

  // write result for this block to global memory
  if(threadIdx.x == 0){
    atomicAddInternal(sum, sdata[0]);
  }
}

//
// PLACE GPU KERNELS HERE - END
//

int main( int argc, char* argv[] ) {
  //
  // Determine min, max, mean, mode and standard deviation of array
  //
  unsigned int array_size, seed, i;
  struct timeval start, end;
  double runtime;

  if( argc < 3 ) {
    printf( "Format: stats_gpu <size of array> <random seed>\n" );
    printf( "Arguments:\n" );
    printf( "  size of array - This is the size of the array to be generated and processed\n" );
    printf( "  random seed   - This integer will be used to seed the random number\n" );
    printf( "                  generator that will generate the contents of the array\n" );
    printf( "                  to be processed\n" );

    exit( 1 );
  }

  //
  // Get the size of the array to process.
  //
  array_size = atoi( argv[1] );

  //
  // Get the seed to be used
  //
  seed = atoi( argv[2] );

  //
  // Make sure that CUDA resources get cleaned up on exit.
  //
  atexit( cleanupCuda );

  //
  // Record the start time.
  //
  gettimeofday( &start, NULL );

  //
  // Allocate the array to be populated.
  //
  double *array = (double *) malloc( array_size * sizeof( double ) );

  //
  // Seed the random number generator and populate the array with its values.
  //
  srand( seed );
  for( i = 0; i < array_size; i++ )
    array[i] = ( (double) rand() / (double) RAND_MAX ) * MAXIMUM_VALUE;

  //
  // Setup output variables to hold min, max, mean, and standard deviation
  //
  // YOUR CALCULATIONS BELOW SHOULD POPULATE THESE WITH RESULTS
  //
  double min = DBL_MAX;
  double max = 0;
  double sum = 0;
  double mean = 0;
  double stddev = 0;

  //
  // CALCULATE VALUES FOR MIN, MAX, MEAN, and STDDEV - BEGIN
  //

  double sqdiffsum = 0;
  double *dev_array, *dev_sum, *dev_min, *dev_max, *dev_sqdiffsum, *dev_mean;
  double size = array_size * sizeof(double);

  cudaMalloc((void**)&dev_array, size);
  cudaMalloc((void**)&dev_sum, sizeof(double));
  cudaMalloc((void**)&dev_min, sizeof(double));
  cudaMalloc((void**)&dev_max, sizeof(double));
  cudaMalloc((void**)&dev_sqdiffsum, sizeof(double));
  cudaMalloc((void**)&dev_mean, sizeof(double));

  cudaMemcpy(dev_array, array, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_sum, &sum, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_min, &min, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_max, &max, sizeof(double), cudaMemcpyHostToDevice);

  dim3 blocksize(256); // create 1D threadblock
  int extra = 0;
  if(array_size % blocksize.x > 0){
    extra = 1;
  }
  dim3 gridsize(array_size/blocksize.x + extra); // create 1D grid, make one more if needed

  reduce<<<gridsize, blocksize>>>(dev_array, dev_sum, dev_min, dev_max, array_size);

  cudaMemcpy(&sum, dev_sum, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&min, dev_min, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&max, dev_max, sizeof(double), cudaMemcpyDeviceToHost);

  mean = sum/array_size;

  cudaMemcpy(dev_array, array, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_sqdiffsum, &sqdiffsum, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_mean, &mean, sizeof(double), cudaMemcpyHostToDevice);

  reducesquarediff<<<gridsize, blocksize>>>(dev_array, dev_sqdiffsum, dev_mean, array_size);

  cudaMemcpy(&sqdiffsum, dev_sqdiffsum, sizeof(double), cudaMemcpyDeviceToHost);

  stddev = sqrt(sqdiffsum / array_size);

  //
  // CALCULATE VALUES FOR MIN, MAX, MEAN, and STDDEV - END
  //

  //
  // Record the end time.
  //
  gettimeofday( &end, NULL );

  //
  // Calculate the runtime.
  //
  runtime = ( ( end.tv_sec  - start.tv_sec ) * 1000.0 ) + ( ( end.tv_usec - start.tv_usec ) / 1000.0 );

  //
  // Output discoveries from the array.
  //
  printf( "Statistics for array ( %d, %d ):\n", array_size, seed );
  printf( "    Minimum = %4.6f, Maximum = %4.6f\n", min, max );
  printf( "    Mean = %4.6f, Standard Deviation = %4.6f\n", mean, stddev );
  printf( "Processing Time: %4.4f milliseconds\n", runtime );

  //
  // Free the allocated array.
  //
  free( array );
  cudaFree(dev_array);

  return 0;
}
