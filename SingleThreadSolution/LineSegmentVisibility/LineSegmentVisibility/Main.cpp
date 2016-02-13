#include <iostream>
#include "DataGenerator.h"
#include <algorithm>
#include <chrono>

using namespace std;
typedef std::chrono::steady_clock Clock;

float *vidnaVisina(lines *data, int N) {
	// Allocate array for visibile section heights
	float* visibileHeight = (float*)malloc(N * sizeof(float));

	// Calculate first line y/x ratio
	float k = data->y[0] / data->x[0];

	// Initialize starting sightHeight
	visibileHeight[0] = data->y[0];
	float *kas = (float*)malloc(N * sizeof(float));
	kas[0] = k;
	for (int i = 1; i < N; i++) {
		
		float prevK = k;
		k = max(k, data->y[i] / data->x[i]);
		kas[i] = k;
		// If new y/x ratio is higher than last one, it means that this line is visibile
		if (k > prevK)
			visibileHeight[i] = data->y[i] - prevK * data->x[i];
		else
			visibileHeight[i] = 0;
	}

	return kas;
}


int main() {

	//initalize random seed
	srand(314);


	float maxAngle = 85.0;
	float p = 0.0;

	//Test your algorithm on the following cases
	//You can also try your own depending on the hardware you have on your disposal
	long long N = (long long)32 * pow((float)10, 6); 		p = 0.01;
	//long long N = (long long)64 * pow((float)10, 6);		p = 0.006;
	//long long N = (long long)128 * pow((float)10, 6);		p = 0.004;
	//long long N = (long long)256 * pow((float)10, 6);		p = 0.002;
	//long long N = (long long)512 * pow((float)10, 6);		p = 0.001;
	//long long N = (long long)1024 * pow((float)10, 6);	p = 0.001;
	//Generate data
	lines data = generateData(N, p, maxAngle);

	//Size of data in MB
	printf("Data size: %d MB\n", N * 2 * sizeof(float) / 1024 / 1024);

	auto t1 = Clock::now();

	float *rez = vidnaVisina(&data, N);

	auto t2 = Clock::now();

	float sum = 0;
	for (int i = 0; i < N; i++) {
		sum += rez[i];
	}

	cout << "Sum: " << sum << endl;
	std::cout << "Time required: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0 << "s" << endl;

	system("pause");

	return 0;
}