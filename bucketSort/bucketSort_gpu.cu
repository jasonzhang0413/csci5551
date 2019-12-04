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

#define TARGET_BUCKET_SIZE  32

#define MAXIMUM_VALUE   1000000.0f
//#define bucketLength
int bucketLength;

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

//
// PLACE GPU KERNELS HERE - BEGIN
//
__device__ void selection_sort(float *data, int size)
{
    for (int i = 0 ; i < size ; ++i)
    {
        float min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j < size ; ++j)
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

__global__ void bucketSortKernel(float *inData, int size, float *outData, int bucketNumber)
{
	__shared__ float **localBucket;
  	__shared__ int *localBucketSize; /* Counter to track the size of each bucket */
	__shared__ int *localCount; /* Counter to track index of each bucket */
  	__shared__ int outIndex;

	int tid = threadIdx.x;
	int offset = blockDim.x;
	int bucket, index;

	if (tid == 0) {
    	localBucket = (float **)malloc(sizeof(float) * bucketNumber);
    	localBucketSize = (int *)malloc(sizeof(float) * bucketNumber);
    	localCount = (int *)malloc(sizeof(float) * bucketNumber);
    	outIndex = 0;
  	}

  	__syncthreads();

  	/* Traverses through the array and calculate each bucket size */
	while(tid < size) {
		bucket = inData[tid] * bucketNumber / MAXIMUM_VALUE;
		atomicAdd(&localBucketSize[bucket], 1);
		tid += offset;
	}
  	//printf( "tid %d, offset %d, localBucketSize %d\n", tid, offset, localBucketSize);
	__syncthreads();

  	tid = threadIdx.x;
  	for (int i=0; i < bucketNumber; i++) {
    	if (tid == i) {
      		localBucket[i] = (float *)malloc(sizeof(float) * localBucketSize[i]);
    	}
  	}

  	__syncthreads();

	/* Traverses through the array and put element into buckets accordingly */
	while(tid < size) {
		bucket = inData[tid] * bucketNumber / MAXIMUM_VALUE;
		index = atomicAdd(&localCount[bucket], 1);
		localBucket[bucket][index] = inData[tid];
		tid += offset;
	}

	__syncthreads();

  	tid = threadIdx.x;
  	// sort each bucket
  	for (int i=0; i < bucketNumber; i++) {
    	if (tid == i) {
      		selection_sort(localBucket[i], localCount[i]);
    	}
  	}

  	__syncthreads();
  	// put each bucket elements into output in order
  	for (int i=0; i < bucketNumber; i++) {
    	if (i == tid) {
      		//printf( "i %d, bucket %d, tid %d, offset %d, localBucketSize %d, localCount %d\n", i, bucket, tid, offset, localBucketSize[i], localCount[i]);
      		for (int j=0; j < localCount[i]; j++) {
        		//printf( "i %d, j %d, index %d, bucket %d, tid %d, offset %d, localBucketSize %d, localCount %d\n", i, j, index, bucket, tid, offset, localBucketSize[i], localCount[i]);
        		outData[outIndex] = localBucket[i][j];
        		atomicAdd(&outIndex, 1);
      		}
    	}
    	__syncthreads();
  	}

}

//
// PLACE GPU KERNELS HERE - END
//

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
  float *array, *results;
  float *d_input, *d_output;

  unsigned int array_size, seed, i, bucketNumber;
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
  array = (float *) malloc( array_size * sizeof( float ) );

  //
  // Seed the random number generator and populate the array with its values.
  //
  srand( seed );
  for( i = 0; i < array_size; i++ )
    array[i] = ( (float) rand() / (float) RAND_MAX ) * MAXIMUM_VALUE;


  /* Number of bucket needed */
  bucketNumber = array_size / TARGET_BUCKET_SIZE;

  results = (float *)malloc(sizeof(float) * array_size);

  // Allocate GPU memory.
  cudaMalloc((void**)&d_input, sizeof(float) * array_size);
  cudaMalloc((void **)&d_output, sizeof(float) * array_size);
  cudaMemset(d_output, 0, sizeof(float) * array_size);

  cudaMemcpy(d_input, array, sizeof(float) * array_size, cudaMemcpyHostToDevice);

  // Execute
  //bucketSortKernel<<<bucketNumber, 1>>>(d_input, array_size, d_output, bucketLength, bucketNumber);
  bucketSortKernel<<<1, bucketNumber>>>(d_input, array_size, d_output, bucketNumber);
  cudaMemcpy(results, d_output, sizeof(float) * array_size, cudaMemcpyDeviceToHost);
  //checkCudaErrors(cudaMemcpy(results, dev_array, array_size*sizeof(float), cudaMemcpyDeviceToHost));

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
  printf( "\n------------\n" );
  printf( "Processing Time: %4.4f milliseconds\n", runtime );

  //
  // Free the allocated array.
  //
  cudaFree(d_input);
  cudaFree(d_output);
  free(array);
  free(results);

  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaDeviceReset();

  return 0;
}
