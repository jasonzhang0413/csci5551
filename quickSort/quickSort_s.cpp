/* C++ implementation of QuickSort */
//#include <bits/stdc++.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#include <cuda.h>

#define MAXIMUM_VALUE   1000000.0f

// A utility function to swap two elements
void swap(float* a, float* b)
{
	float t = *a;
	*a = *b;
	*b = t;
}

/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
array, and places all smaller (smaller than pivot)
to left of pivot and all greater elements to right
of pivot */
int partition (float arr[], int low, int high)
{
	float pivot = arr[high]; // pivot
	int i = low; // Index of pivot element

	for (int j = low; j < high; j++)
	{
		// If current element is smaller than the pivot
		if (arr[j] < pivot)
		{
			swap(&arr[i], &arr[j]);
			i++; // increment index of pivot element
		}
	}
	swap(&arr[i], &arr[high]);
	return i;
}

/* The main function that implements QuickSort
arr[] --> Array to be sorted,
low --> Starting index,
high --> Ending index */
void quickSort(float arr[], int low, int high)
{
	if (low < high)
	{
		/* pi is partitioning index, arr[p] is now
		at right place */
		int pi = partition(arr, low, high);

		// Separately sort elements before
		// partition and after partition
		quickSort(arr, low, pi - 1);
		quickSort(arr, pi + 1, high);
	}
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

// Driver Code
int main(int argc, char* argv[])
{
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


  quickSort(array, 0, array_size - 1);

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
