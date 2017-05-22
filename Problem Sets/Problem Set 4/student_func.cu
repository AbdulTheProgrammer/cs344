//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
__global__ 
void makeHisto(unsigned int* histo, unsigned int* vals, unsigned int size, unsigned int mask) {
	int myId = threadIdx.x + blockIdx.x * blockDim.x;
	if (myId < size) {
		int bin = ((vals[myId] & mask) > 0) ? 1 : 0;
		histo[bin]++; 
	}
}
__global__
void scanHist(unsigned int* d_histo, unsigned int size) {
	//assuming 1 bit keys 
	//TODO: generalize to n bit keys
	int myId = threadIdx.x + blockIdx.x * blockDim.x;
	if (myId)
		d_histo[1] += d_histo[0];
}
__global__ 
void generatePredicateArrays(unsigned int* vals, unsigned int* true_predicates, unsigned int* false_predicates, unsigned int mask, unsigned int numElems) {
	int myId = threadIdx.x + blockIdx.x * blockDim.x; 
	if (myId < numElems) {
		if (!(vals[myId] & mask)) {
			true_predicates[myId] = 1; 
			false_predicates[myId] = 0; 
		}
		else {
			true_predicates[myId] = 0;
			false_predicates[myId] = 1;
		}
	}
}

__global__
void generateOffsets(unsigned int *d_truePs, unsigned int *d_falsePs, unsigned int * d_sumTruePs, unsigned int * d_sumFalsePs,  unsigned int *d_aux_1, unsigned int *d_aux_2, unsigned int size) {
	extern __shared__ unsigned int shared[];
	
	unsigned int * shared_1 = shared; 
	unsigned int * shared_2 = (unsigned int * )(&(shared[blockDim.x*2]));

	unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	int pout = 0, pin = 1;

	shared_1[blockDim.x*pout + tid] = (tid == 0) ? 0 : d_truePs[myId - 1];
	shared_2[blockDim.x*pout + tid] = (tid == 0) ? 0 : d_falsePs[myId - 1];

	if (tid == blockDim.x - 1) {
		d_aux_1[blockIdx.x] = d_truePs[myId];
		d_aux_2[blockIdx.x] = d_falsePs[myId];
	}
	__syncthreads();
	
	for (int offset = 1; offset <= blockDim.x; offset <<= 1) {
		pout = !pout;
		pin = !pin;

		shared_1[pout*blockDim.x + tid] = shared_1[pin*blockDim.x + tid];
		shared_2[pout*blockDim.x + tid] = shared_2[pin*blockDim.x + tid];
		if (tid - offset >= 0) {
			shared_1[pout*blockDim.x + tid] += shared_1[pin*blockDim.x + tid - offset];
			shared_2[pout*blockDim.x + tid] += shared_2[pin*blockDim.x + tid - offset];
		}
		__syncthreads();
	}
	
	d_sumTruePs[myId] = shared_1[pout*blockDim.x + tid];
	d_sumFalsePs[myId] = shared_2[pout*blockDim.x + tid];
	if (tid == blockDim.x - 1) {
		d_aux_1[blockIdx.x] += shared_1[pout*blockDim.x + tid];
		d_aux_2[blockIdx.x] += shared_2[pout*blockDim.x + tid];
	} 
}

__global__
void add_sums(unsigned int *d_truePs, unsigned int *d_falsePs, unsigned int *d_aux_1, unsigned int *d_aux_2) {
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	d_truePs[myId] += d_aux_1[blockIdx.x];
	d_falsePs[myId] += d_aux_2[blockIdx.x];
}
__global__
void applyOffsets(unsigned int * d_trueP, unsigned int * d_falseP,unsigned int * d_sumTrueP , unsigned int * d_sumFalseP,  unsigned int * d_histo, unsigned int * d_outputVals, unsigned int *d_inputVals, unsigned int size) {
	int myId = threadIdx.x + blockDim.x * blockIdx.x;  
	if (myId < size) {
		if (d_trueP[myId]) {
			d_outputVals[d_histo[0] + d_sumTrueP[myId]] = d_inputVals[myId];
		}
		else {
			d_outputVals[d_histo[1] + d_sumFalseP[myId]] = d_inputVals[myId];
		}
	}
}

__global__
void set(float * dst) {
	dst[threadIdx.x] = 1;
}
__global__
void copy(unsigned int * dst, const unsigned * const src, int size) {
	int myId = blockDim.x * blockIdx.x + threadIdx.x;
	if (myId < size) {
		dst[myId] = src[myId];
	}
}
__global__
void pad(unsigned int * dst, const int start, const unsigned int padding, unsigned  int size) {
	int myId = blockDim.x * blockIdx.x + threadIdx.x;
	if (start + myId < size) {
		dst[start + threadIdx.x] = padding;
	}
}

bool IsPowerOfTwo(unsigned long x) {
	return x != 0 && ((x & (x - 1)) == 0);
}

int nearestPowerOfTwoUpper(unsigned long x) {
	return static_cast<int> (pow(2, ceil(log(x) / log(2))));
}
int nearestPowerOfTwoLower(unsigned long x) {
	return static_cast<int> (pow(2, floor(log(x) / log(2))));
}

unsigned int * resize_d_array(unsigned int * d_array, unsigned int old_num, unsigned int new_num, unsigned int padding_num) {
	if (old_num == new_num)
		return d_array;
	unsigned int * new_array;
	checkCudaErrors(cudaMalloc(&new_array, sizeof(unsigned int) * new_num));
	int threads = ceil(new_num / 32); 
	int blocks = 32;
	copy << <blocks, threads >> >(new_array, d_array, new_num);
	if (new_num > old_num ) {
		pad << <blocks, threads >> >(new_array, old_num - 1, padding_num, new_num);
	}
	return new_array;
}

void swap(unsigned int ** a, unsigned int ** b) {
	unsigned int * temp = *a; 
	*a = *b; 
	*b = temp;
}
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
	
	printf("num elements = %i \n", numElems);
	int numOfBits = 1;
	int numOfBins = 1 << numOfBits; 
	unsigned int* d_tempInputVals = d_inputVals;
	unsigned int* d_tempOutputVals = d_inputVals;
	//unsigned int* d_tempInputPos = d_inputPos; 
	//unsigned int* d_tempOutputPos = d_outputPos; 
	unsigned int* d_truePredicateArray; 
	unsigned int* d_falsePredicateArray; 
	unsigned int* d_resizedTruePredicates; 
	unsigned int* d_resizedSummedFalsePredicates; 
	unsigned int* d_resizedSummedTruePredicates;
	unsigned int* d_summedFalsePredicates;
	unsigned int* d_summedTruePredicates;
	unsigned int* d_resizedFalsePredicates;
	unsigned int* d_aux_array1;
	unsigned int* d_aux_array2; 
	
	int threads = ceil(numElems / 32.);
	int blocks = 32;
	int new_size = nearestPowerOfTwoUpper(numElems);
	unsigned int* d_histo; 

	checkCudaErrors(cudaMalloc(&d_summedFalsePredicates, sizeof(unsigned int)*numElems));
	checkCudaErrors(cudaMalloc(&d_summedTruePredicates, sizeof(unsigned int)*numElems));
	checkCudaErrors(cudaMalloc(&d_aux_array1, sizeof(unsigned int)*blocks));
	checkCudaErrors(cudaMalloc(&d_aux_array2, sizeof(unsigned int)*blocks));
	checkCudaErrors(cudaMalloc(&d_truePredicateArray, sizeof(unsigned int)*numElems));
	checkCudaErrors(cudaMalloc(&d_falsePredicateArray, sizeof(unsigned int)*numElems));
	checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int)*numOfBins));
	
	checkCudaErrors(cudaMalloc(&d_resizedTruePredicates, sizeof(unsigned int)*new_size));
	checkCudaErrors(cudaMalloc(&d_resizedFalsePredicates, sizeof(unsigned int)*new_size));
	checkCudaErrors(cudaMalloc(&d_resizedSummedFalsePredicates, sizeof(unsigned int)*new_size));
	checkCudaErrors(cudaMalloc(&d_resizedSummedTruePredicates, sizeof(unsigned int)*new_size));

	unsigned int* h_output = new unsigned int[new_size];
	unsigned int *h_bin = new unsigned int[numOfBins];
	unsigned int *h_aux = new unsigned int[blocks];
	int mask;

	for (unsigned int i = 0; i < 8*sizeof(unsigned int); i += numOfBits) {
		mask = 1 << i;
		makeHisto << <blocks, threads >> >(d_histo, d_tempInputVals, numElems, mask);
		scanHist << <1, numOfBins >> > (d_histo, numOfBins);
		
		generatePredicateArrays<<<blocks, threads>>>(d_tempInputVals, d_truePredicateArray, d_falsePredicateArray, mask, numElems); 
		
		//assuming numElems not a power of two 
		copy << <blocks, threads >> > (d_resizedFalsePredicates, d_falsePredicateArray, numElems);
		copy << <blocks, threads >> > (d_resizedTruePredicates, d_truePredicateArray, numElems);
		threads = ceil(new_size / 32); 
		generateOffsets << <blocks, threads , threads * 4 * sizeof(unsigned int)>> >(d_resizedTruePredicates, d_resizedFalsePredicates, d_resizedSummedTruePredicates, d_resizedSummedFalsePredicates, d_aux_array1, d_aux_array2, new_size);
		add_sums << <blocks, threads>> >(d_resizedSummedTruePredicates, d_resizedSummedFalsePredicates, d_aux_array1, d_aux_array2);
		cudaDeviceSynchronize();
		if (i == 0) {
			checkCudaErrors(cudaMemcpy(h_output, d_resizedSummedFalsePredicates, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(h_aux, d_aux_array2, blocks*sizeof(unsigned int), cudaMemcpyDeviceToHost));
			/*for (int k = 0; k < blocks; k++) {
				printf("%u %u \n", h_aux[k], blocks);
			}*/
			for (int k = 0; k < numElems; k++) {
				printf("%u %u %i %i \n", h_output[k], blocks, new_size, numElems);
			}
		}
		copy << <blocks, threads >> > (d_summedFalsePredicates,d_resizedSummedFalsePredicates ,numElems);
		copy << <blocks, threads >> > (d_summedTruePredicates,d_resizedSummedTruePredicates ,numElems);
		applyOffsets << <blocks, threads >> > (d_truePredicateArray, d_falsePredicateArray,d_summedTruePredicates, d_summedFalsePredicates, d_histo, d_tempOutputVals, d_tempInputVals, numElems);
		swap(&d_tempOutputVals, &d_tempInputVals);
	}
	copy<<<blocks, threads>>>(d_outputVals, d_tempInputVals, numElems);
	/**/
	/*	1) create a histogram from the bins 
		2) scan the histogram 
		4) Create offset from predicates

		3) place elements in the correct locations using a scatter operation 
	*/
}
