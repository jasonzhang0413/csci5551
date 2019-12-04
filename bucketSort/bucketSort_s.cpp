// C++ program to sort an array using bucket sort
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
using namespace std;

#define MAXIMUM_VALUE   1000000.0f

// Function to sort arr[] of size n using bucket sort
void bucketSort(float arr[], int n, int bucketNumber)
{
	// 1) Create n empty buckets
	vector<float> b[bucketNumber];

	// 2) Put array elements in different buckets
	for (int i=0; i<n; i++)
	{
		int bi = bucketNumber*arr[i]/MAXIMUM_VALUE; // Index in bucket
		b[bi].push_back(arr[i]);
	}

	// 3) Sort individual buckets
	for (int i=0; i<bucketNumber; i++)
		sort(b[i].begin(), b[i].end());

	// 4) Concatenate all buckets into arr[]
	int index = 0;
	for (int i = 0; i < bucketNumber; i++)
		for (int j = 0; j < b[i].size(); j++)
			arr[index++] = b[i][j];
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

int main(int argc, char* argv[])
{
	unsigned int array_size, seed, i, bucketNumber;
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


	//printArray(array, array_size);

	bucketNumber = array_size / 32;

	printf( "bucketNumber for array is ( %d ):\n", bucketNumber );

	bucketSort(array, array_size, bucketNumber);

	//
  	// Record the end time.
  	//
  	gettimeofday( &end, NULL );

  	//
  	// Calculate the runtime.
  	//
  	runtime = ( ( end.tv_sec  - start.tv_sec ) * 1000.0 ) + ( ( end.tv_usec - start.tv_usec ) / 1000.0 );

	printf( "Statistics for array ( %d, %d ):\n", array_size, seed );
	//printArray(array, array_size);
	checkResult(array, array_size);
	printf( "Processing Time: %4.4f milliseconds\n", runtime );
	return 0;

}
