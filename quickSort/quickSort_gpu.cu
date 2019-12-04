#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <helper_cuda.h>
#include <helper_string.h>

#define MAX_DEPTH       24
#define INSERTION_SORT  32

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

__device__ float device_pow( float x, float y ) {
  //
  // Calculate x^y on the GPU.
  //
  return pow( x, y );
}

//
// PLACE GPU KERNELS HERE - BEGIN
//

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(float *data, int left, int right)
{
    for (int i = left ; i <= right ; ++i)
    {
        float min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j <= right ; ++j)
        {
            float val_j = data[j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(float *data, int left, int right, int depth)
{
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT)
    {
        selection_sort(data, left, right);
        return;
    }

    float *lptr = data+left;
    float *rptr = data+right;
    //float  pivot = data[(left+right)/2];
    float  pivot = data[right];

    // Do the partitioning.
    while (lptr <= rptr)
    {
        // Find the next left- and right-hand values to swap
        unsigned int lval = *lptr;
        unsigned int rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot)
        {
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot)
        {
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr)
        {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr-data))
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}

//
// PLACE GPU KERNELS HERE - END
//

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(float *data, unsigned int nitems)
{
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    // Launch on device
    int left = 0;
    int right = nitems-1;
    std::cout << "Launching kernel on the GPU" << std::endl;
    cdp_simple_quicksort<<< 1, 1 >>>(data, left, right, 0);
    checkCudaErrors(cudaDeviceSynchronize());
}

void printArray(float arr[], int size)
{
	int i;
	printf( "array size is %d\n", size);
	for (i = 0; i < size; i++)
		printf( "%f ", arr[i]);
}

void checkResult(float array[], int size)
{
	float temp = 0;
  bool checkResult = true;
  for (int i=0; i < size; i++) {
    if (temp > array[i]) {
      checkResult = false;
      break;
    }
    temp = array[i];
  }
  if (checkResult) {
    printf( "Result sorted correct\n");
  } else {
    printf( "Result sorted wrong\n");
  }
}

int main( int argc, char* argv[] ) {
  //
  // Determine min, max, mean, mode and standard deviation of array
  //
  unsigned int array_size, seed, i;
  struct timeval start, end;
  double runtime;

  if( argc < 3 ) {
    printf( "Format: quickSort_gpu <size of array> <random seed>\n" );
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
  float *array = (float *) malloc( array_size * sizeof( float ) );

  //
  // Seed the random number generator and populate the array with its values.
  //
  srand( seed );
  for( i = 0; i < array_size; i++ )
    array[i] = ( (float) rand() / (float) RAND_MAX ) * MAXIMUM_VALUE;


  float *dev_array;
  // Allocate GPU memory.
  checkCudaErrors(cudaMalloc((void **)&dev_array, array_size * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dev_array, array, array_size * sizeof(float), cudaMemcpyHostToDevice));

  // Execute
  run_qsort(dev_array, array_size);

  float *results = new float[array_size];
  checkCudaErrors(cudaMemcpy(results, dev_array, array_size*sizeof(float), cudaMemcpyDeviceToHost));

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
  //printArray(array, array_size);
  printf( "\n------------\n" );
  //printArray(results, array_size);
  checkResult(results, array_size);
  printf( "Processing Time: %4.4f milliseconds\n", runtime );

  //
  // Free the allocated array.
  //
  free( array );
  cudaFree(dev_array);

  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaDeviceReset();

  return 0;
}
