/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdint.h>
#include <float.h>

__device__ float dev_min(float a, float b) {
	return (a > b) ? b : a;
}
__device__ float dev_max(float a, float b) {
	return (a > b) ? a : b;
}

__global__
void parallel_minmax(const float* const d_in, float * min_d_out, float * max_d_out, const size_t numCols, const size_t numRows) {

	extern __shared__ float shared_array[];
		
	float *max_sdata = shared_array;
	float *min_sdata = &(shared_array[blockDim.x]);

	int myId = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;

	//if (myId < numRows*numCols)
	if(min_d_out)
		min_sdata[tid] = d_in[myId];
	if(max_d_out)
		max_sdata[tid] = d_in[myId];
	
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			if(min_d_out)
				min_sdata[tid] = dev_min(min_sdata[tid], min_sdata[tid + s]);
			if(max_d_out)
				max_sdata[tid] = dev_max(max_sdata[tid], max_sdata[tid + s]);
		}
		__syncthreads();
	}
	if (tid == 0) {
		if(min_d_out)
			min_d_out[blockIdx.x] = min_sdata[tid];
		if(max_d_out)
			max_d_out[blockIdx.x] = max_sdata[tid];
	}
}
__global__
void scan_large(unsigned int *d_out, unsigned int *d_in, unsigned int * d_aux,  unsigned int size) {
	extern __shared__ int shared[];
	
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x; 
	int pout = 0, pin = 1;

	shared[blockDim.x*pout + tid] = (tid == 0) ? 0 : d_in[myId - 1];
	if (tid == blockDim.x - 1) {
		d_aux[blockIdx.x] = d_in[myId]; 
	}
	__syncthreads();

	for (int offset = 1; offset <= blockDim.x; offset <<= 1) {
		pout = !pout;
		pin = !pin;
		shared[pout*blockDim.x + tid] = shared[pin*blockDim.x + tid];
		if (tid - offset >= 0) {
			shared[pout*blockDim.x + tid] += shared[pin*blockDim.x + tid - offset];
		}

		__syncthreads();
	}

	d_out[myId] = shared[pout*blockDim.x + tid];
	if (tid == blockDim.x - 1) {
		d_aux[blockIdx.x] += shared[pout*blockDim.x + tid];
	}
}

__global__ 
void add_all(unsigned int *d_aux, unsigned int *d_out) {
	int myId = threadIdx.x + blockDim.x * blockIdx.x; 
	d_out[myId] += d_aux[blockIdx.x]; 
}

__global__ 
void scan(unsigned *d_out, unsigned *d_in, int size) {
	extern __shared__ int shared[]; 
	int tid = threadIdx.x; 
	int pout = 0, pin = 1;

	shared[size*pout + tid] = (tid == 0) ? 0 : d_in[tid - 1]; 
	__syncthreads(); 
	//int val; 
	for (int offset = 1; offset <= size; offset <<= 1) {
		pout = !pout; 
		pin = !pin; 
		shared[pout*size + tid] = shared[pin*size + tid];
		if (tid - offset >= 0) {
			shared[pout*size + tid] += shared[pin*size + tid - offset];
		} 
		

		/*if (tid - offset >= 0)
			val = d_out[tid - offset];
		__syncthreads(); 
		if (tid - offset >= 0)
			d_out[tid - offset] += val;*/
		__syncthreads();
	}

	d_out[tid] = shared[pout*size + tid];
}
__global__
void generate_histogram(const float* const d_logLuminance, unsigned * d_histo, const int numOfBins, const float range, const float min) {
	//extern __shared__ float shared_array[];
	int id = threadIdx.x + blockDim.x * blockIdx.x; 
	int bin = (d_logLuminance[id] - min) * (numOfBins) / range; 
	atomicAdd(&(d_histo[bin]), 1);
}
__global__
void set(float * dst) {
	dst[threadIdx.x] = 1;
}
__global__ 
void copy(float * dst, const float * const src) {
	dst[threadIdx.x] = src[threadIdx.x];
}

__global__
void pad(float * dst, const int start, const float padding) {
	dst[start + threadIdx.x] = padding;
}

bool IsPowerOfTwo(unsigned long x) {
	return x != 0 && ((x & (x - 1)) == 0);
}

int nearestPowerOfTwo(unsigned long x) {
	return static_cast<int> (pow(2,ceil(log(x)/log(2))));
}

float * resize_d_array(float * d_array, int old_num ,int new_num, float padding_num) {
	if (old_num == new_num)
		return d_array;
	float * new_array; 
	checkCudaErrors(cudaMalloc(&new_array, sizeof(float) * new_num));
	copy <<<1, new_num >>>(new_array, d_array);
	if (new_num > old_num) {
		pad <<<1, new_num - old_num >>>(new_array, old_num - 1, padding_num);
	}
	checkCudaErrors(cudaFree(d_array));
	return new_array;
}
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
	// step 1: calculate max and min using the a parallel reduce strategy 
	int thread_num = 1024;
	int block_num = (numCols*numRows) / thread_num ;
	//TODO handle the case if the dimensions of the picture are not divisible evenly by thread_num (add padding to d_logLum)
	float * d_intermediate_min;
	float * d_intermediate_max;
	float * d_max;
	float * d_min;
	checkCudaErrors(cudaMalloc((void **) &d_intermediate_min, sizeof(float) * block_num ));
	checkCudaErrors(cudaMalloc((void **) &d_intermediate_max, sizeof(float) * block_num));
	checkCudaErrors(cudaMalloc((void **) &d_min, sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_max, sizeof(float)));

	parallel_minmax <<<block_num, thread_num, thread_num * sizeof(float) * 2 >>>(d_logLuminance, d_intermediate_min, d_intermediate_max, numCols, numRows);
	if (!IsPowerOfTwo(block_num)) {
		int temp = nearestPowerOfTwo(block_num);
		printf("%i \n", temp);
		d_intermediate_min = resize_d_array(d_intermediate_min,block_num,temp, FLT_MAX);
		d_intermediate_max = resize_d_array(d_intermediate_max, block_num, temp, -FLT_MAX);
		block_num = temp;
	}
	float *h_intermediate_max = new float[block_num];
	checkCudaErrors(cudaMemcpy(h_intermediate_max, d_intermediate_max,sizeof(float) * block_num, cudaMemcpyDeviceToHost));
	for (int i = 0; i < block_num; i++)
		printf("this is the val %f %i \n", h_intermediate_max[i], block_num);
	thread_num = block_num; 
	block_num = 1;
	parallel_minmax <<<block_num, thread_num, thread_num * sizeof(float) * 2 >>>(d_intermediate_min, d_min, NULL, numCols, numRows);
	parallel_minmax <<<block_num, thread_num, thread_num * sizeof(float) * 2 >>>(d_intermediate_max, NULL,d_max, numCols, numRows);
	
	checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float) ,cudaMemcpyDeviceToHost));
	printf("This is the max val %f min val %f from CUDA \n",  max_logLum, min_logLum); 
	
	//step 2 calculate the range 
	float range = max_logLum - min_logLum; 

	//step 3: generate the historgram 
	unsigned * d_histo;
	unsigned * h_histo = new unsigned[numBins];
	checkCudaErrors(cudaMalloc(&d_histo, sizeof(int) * numBins));
	checkCudaErrors(cudaMemset(d_histo,0 ,sizeof(int) * numBins));
	thread_num = 1024; 
	block_num = (numCols*numRows) / thread_num;
	generate_histogram <<<thread_num, block_num >>>(d_logLuminance,d_histo, numBins, range, min_logLum );
	thread_num = 32;
	block_num = numBins / 32;
	unsigned * d_aux; 
	// step 4: generate CDF 
	checkCudaErrors(cudaMalloc(&d_aux, sizeof(int) * block_num)); 
	scan_large << <block_num, thread_num, block_num * sizeof(int) * 2 >> >(d_cdf, d_histo, d_aux,  numBins); 
	scan << <1, block_num, block_num * sizeof(int) * 2 >> >(d_aux, d_aux, block_num);
	add_all << <block_num, thread_num >> >(d_aux, d_cdf);
	unsigned * h_cdf = new unsigned[numBins];
	checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, sizeof(int) * numBins, cudaMemcpyDeviceToHost)); 
	/*for (int i = 0; i < numBins; i++) {
		printf("%u \n", h_cdf[i]);
	}*/
	//step 4: perform an exclusive scan of the histogram to obtain the cdf

  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */


}
