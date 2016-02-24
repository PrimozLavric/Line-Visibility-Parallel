#include <iostream>
#include "DataGenerator.h"
#include <algorithm>
#include <chrono>
#include <vector>

using namespace std;
typedef std::chrono::steady_clock Clock;
vector<double> stopwatch;

float *vidnaVisina(lines *data, int N) {
	// Allocate array for visibile section heights
	float* visibileHeight = (float*)malloc(N * sizeof(float));
	float* k = (float*)malloc(N * sizeof(float));

	auto t1 = Clock::now();

	// Calculate first line y/x ratio
	k[0] = data->y[0] / data->x[0];

	// Initialize starting sightHeight
	visibileHeight[0] = data->y[0];

	for (int i = 1; i < N; i++) {
		k[i] = fmax(k[i - 1], data->y[i] / data->x[i]);

		// If new y/x ratio is higher than last one, it means that this line is visibile
		if (k[i] > k[i - 1])
			visibileHeight[i] = data->y[i] - k[i - 1] * data->x[i];
		else
			visibileHeight[i] = 0;
	}

	auto t2 = Clock::now();
	stopwatch.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0);
	free(k);

	return visibileHeight;
}


int main() {

	//initalize random seed
	srand(time(NULL));

	float maxAngle = 85.0;
	float p = 0.0;
	long long N = 1;
	int iter = 100;
	
	vector<long long> Ns;
	vector<float> ps;
	//Test your algorithm on the following cases
	//You can also try your own depending on the hardware you have on your disposal
	
	N = (long long)512 * pow((float)10, 6);		p = 0.001;
	Ns.push_back(N); ps.push_back(p);
	N = (long long)256 * pow((float)10, 6);		p = 0.002;
	Ns.push_back(N); ps.push_back(p);
	N = (long long)128 * pow((float)10, 6);		p = 0.004;
	Ns.push_back(N); ps.push_back(p);
	N = (long long)64 * pow((float)10, 6);		p = 0.006;
	Ns.push_back(N); ps.push_back(p);
	N = (long long)32 * pow((float)10, 6); 		p = 0.01;
	Ns.push_back(N); ps.push_back(p);
	

	for (int i = 0; i < Ns.size(); i++) {
		// generate data
		lines data = generateData(Ns[i], ps[i], maxAngle);
		//Size of data in MB
		printf("Data size: %d MB\n", Ns[i] * 2 * sizeof(float) / 1024 / 1024);

		for (int j = 0; j < iter; j++) {
			float *rez = vidnaVisina(&data, Ns[i]);
			free(rez);
		}

		// Mean
		double mean = 0;
		for (int i = 0; i < iter; i++) {
			mean += stopwatch[i];
		}
		mean = mean / iter;

		// Standard deviation
		double sDev = 0;
		for (int i = 0; i < iter; i++) {
			sDev += pow(stopwatch[i] - mean, 2);
		}
		sDev = sqrt(sDev / iter);

		cout << "No. of iterations: " << iter << endl;
		cout << "Mean: " << mean << endl;
		cout << "Standard deviation: " << sDev << endl << endl;

		free(data.x);
		free(data.y);
		stopwatch.clear();
	}

	system("pause");

	return 0;
}