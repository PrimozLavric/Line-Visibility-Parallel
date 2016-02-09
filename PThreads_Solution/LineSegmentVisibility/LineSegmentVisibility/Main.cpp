#include <iostream>
#include "DataGenerator.h"
#include <algorithm>
#include <chrono>
#include <pthread.h>

#define N_THREAD 1

using namespace std;
typedef std::chrono::steady_clock Clock;

// Thread barrier
pthread_barrier_t barrier;
// Thread args
struct pArgs {
	pArgs(int ID, int threadCount, lines* data, float *k, float *visibileHeights, int size) {
		this->ID = ID;
		this->threadCount = threadCount;
		this->data = data;
		this->k = k;
		this->size = size;
		this->visibileHeights = visibileHeights;
	}
	int ID;
	int threadCount;
	lines* data;
	float *k;
	float *visibileHeights;
	int size;
};

void *calculateVisibility(void *arg) {
	pArgs *args = (pArgs*) arg;
	
	// Read arguments (For the sake of understanding)
	int myID				= args->ID;
	int threadCount			= args->threadCount;
	lines *data				= args->data;
	float *k				= args->k;
	float *visibileHeights	= args->visibileHeights;
	int size				= args->size;


	// ------------------------------------------------------------------
	// --------------------- UPSWEEP (REDUCE) PHASE ---------------------
	// ------------------------------------------------------------------

	for (int i = 0; i < log2(size); i++) {
		// WORK DISTRIBUTION
		
		// Calculate global steps and local steps
		int forwStep	= 1 << i + 1;
		int backStep	= 1 << i;
		int stepsAll	= size / forwStep;
		int stepsThread = stepsAll / threadCount;
		
		// Calculate local begining index
		int begin = (forwStep - 1) + myID * (stepsAll / threadCount) *  forwStep;

		// Check if there is any work remaining
		// Distribute remaining work among first (stepsAll mod threadCount) threads andadjust local begining index
		int remainder = stepsAll % threadCount;
		if (remainder != 0) {
			// Distribute remaining work
			if (remainder - myID > 0)
				stepsThread += 1;

			// Adjust begining index
			begin += min(remainder, myID) * forwStep;
		}
		
		// ACTUAL WORK

		// In first iteration we must also do the initialization of coefficients
		if (i == 0) {
			auto t1 = Clock::now();

			for (int j = begin; j < begin + (stepsThread * forwStep); j += forwStep) {
				// In each iteration we initialize 2 coefficients ==> 2^0 = 1;
				k[j - backStep] = data->y[j - backStep] / data->x[j - backStep];
				k[j] = data->y[j] / data->x[j];

				// Check if line j - 1 completely or just partialy obscures line j
				if (k[j] < k[j - backStep])
					k[j] = k[j - backStep];
			}

			// Time required
			auto t2 = Clock::now();
			std::cout << "Time required: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0 << "s" << endl;
		}
		else {
			for (int j = begin; j < begin + (stepsThread * forwStep); j += forwStep) {
				if (k[j] < k[j - backStep])
					k[j] = k[j - backStep];
			}
		}

		// Synchronize every upsweep iteration
		pthread_barrier_wait(&barrier);
	}

	// ------------------------------------------------------------------
	// -------------------- DOWNSWEEP (REDUCE) PHASE --------------------
	// ------------------------------------------------------------------
	// Variable for storing temporary coefficient in downstep
	float tempK;
	if (myID == 0) {
		k[size-1] = 0;
	}

	for (int i = log2(size) - 1; i >= 0; i--) {
		// WORK DISTRIBUTION

		// Calculate global steps and local steps
		int forwStep = pow(2, i + 1);
		int backStep = pow(2, i);
		int stepsAll = size / forwStep;
		int stepsThread = stepsAll / threadCount;

		// Calculate local begining index
		int begin = (forwStep - 1) + myID * (stepsAll / threadCount) *  forwStep;

		// Check if there is any work remaining
		// Distribute remaining work among first (stepsAll mod threadCount) threads and adjust local begining index
		int remainder = stepsAll % threadCount;
		if (remainder != 0) {
			// Distribute remaining work
			if (remainder - myID > 0)
				stepsThread += 1;

			// Adjust begining index
			begin += min(remainder, myID) * forwStep;
		}

		// ACTUAL WORK
		for (int j = begin; j < begin + (stepsThread * forwStep); j += forwStep) {
			tempK = max(k[j], k[j - backStep]);

			k[j - backStep] = k[j];
			k[j] = tempK;
		}

		// When we are done calculating coefficients.. Calculate actual visibile area
		if (i == 0) {
			for (int j = begin; j < begin + (stepsThread * forwStep); j += forwStep) {
				// If visibile height is < 0.. Round it up to 0
				visibileHeights[j - backStep] = max(data->y[j - backStep] - (data->x[j - backStep] * k[j-backStep]), 0.0f);
				visibileHeights[j] = max(data->y[j] - (data->x[j] * k[j]), 0.0f);
			}
		}

		// Synchronize every downsweep iteration
		pthread_barrier_wait(&barrier);

	}

	return 0;
}


// Calculate visibile line segments
// WARNING: DO NOT PASS STATICALLY ALLOCATED DATA HERE!
float* visibileSegments(lines* data, int size) {
	// Check if we need to expand size to be equal to 2^n where
	int worksize = pow(2, ceil(log2(size * 1.0)));
	if (worksize > size) {
		// Try to reallocate arrays
		data->x = (float *) realloc(data->x, worksize * sizeof(float));
		data->y = (float *) realloc(data->y, worksize * sizeof(float));
		if (data->x == NULL || data->y == NULL) {
			cout << "Realloc failure!" << endl;
			exit(1);
		}

		// Set extra values
		for (int i = size; i < worksize; i++) {
			data->x[i] = FLT_MAX;
			data->y[i] = 0;
		}
	}

	pthread_barrier_init(&barrier, NULL, N_THREAD);

	// Used for thread management
	pthread_t t[N_THREAD];
	
	// Allocate memory for coefficients and results
	float *k = (float*) malloc(worksize * sizeof(float));
	float *visibileHeights = (float*)malloc(worksize * sizeof(*data->x));

	auto t1 = Clock::now();

	// Run threads
	for (int i = 0; i < N_THREAD; i++){
		// Generate thread arguments
		pArgs *args = new pArgs(i, N_THREAD, data, k, visibileHeights, worksize);

		pthread_create(&t[i], NULL, calculateVisibility, args);
	}

	for (int i = 0; i < N_THREAD; i++) {
		pthread_join(t[i], NULL);
	}

	// Time required for parallel processing
	auto t2 = Clock::now();
	std::cout << "Time required: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0 << "s" << endl;

	return visibileHeights;
}


int main() {

	//initalize random seed
	srand(314);


	float maxAngle = 85.0;
	float p = 0.0;

	//Test your algorithm on the following cases
	//You can also try your own depending on the hardware you have on your disposal
	//long long N = (long long)32 * pow((float)10, 6); 		p = 0.01;
	//long long N = (long long)64 * pow((float)10, 6);		p = 0.006;
	//long long N = (long long)128 * pow((float)10, 6);		p = 0.004;
	long long N = (long long)256 * pow((float)10, 6);		p = 0.002;
	//long long N = (long long)512 * pow((float)10, 6);		p = 0.001;
	//long long N = (long long)1024 * pow((float)10, 6);	p = 0.001;

	//Generate data
	lines data = generateData(N, p, maxAngle);

	//Size of data in MB
	printf("Data size: %d MB\n", N * 2 * sizeof(float) / 1024 / 1024);

	float *rez = visibileSegments(&data, N);

	float sum = 0;
	for (int i = 0; i < N; i++) {
		sum += rez[i];
	}

	cout << "Sum: " << sum << endl;
	

	system("pause");

	return 0;
}