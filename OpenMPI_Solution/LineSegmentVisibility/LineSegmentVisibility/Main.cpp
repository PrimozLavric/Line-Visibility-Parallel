#include <iostream>
#include <mpi.h>
#include <string.h> 
#include <stdio.h>
#include <iostream>
#include <time.h> 
#include "DataGenerator.h"
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

vector<double> stopwatch;
#define NUM_ITER 2

int main(int argc, char*argv[]) {
	// MPI VARS
	int my_id;				// Process id
	int num_of_processes;	// Process count
	int destination;		// Receivers rank
	MPI_Status status;		// Return status
	int* displs = 0;		// Displacements array
	int* sendcnts = 0;		// Number of elements that are to be sent
	float* localX = 0;		// Local process X value pointer
	float* localK = 0;		// Local coefficients
	float* localY = 0;		// Local process Y value pointer
	int tag = 0;
	double t1, t2;

	// PROBLEM VARS
	lines data;
	float *results = 0;
	data.x = 0;
	data.y = 0;
	long long N;

	// MPI initialization
	MPI_Init(&argc, &argv);

	// Fetch current process rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	// Fetch process count
	MPI_Comm_size(MPI_COMM_WORLD, &num_of_processes);

	float maxAngle = 85.0;
	float p = 0.0;
	
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

	ofstream myfile;

	stringstream strStream;
	strStream << "Result_" << num_of_processes << ".txt";

	myfile.open(strStream.str().c_str());

	for (int i = 0; i < Ns.size(); i++) {
		// Process with rank 0 should generate the data
		if (my_id == 0) {
			// Initalize random seed
			srand(time(NULL));

			// Actual data generation
			data = generateData(Ns[i], ps[i], maxAngle);
			results = (float *)malloc(Ns[i] * sizeof(float));
			myfile << "Data size: " << Ns[i] * 2 * sizeof(float) / 1024 / 1024 << " MB" << endl;
		}

		for (int j = 0; j < NUM_ITER; j++) {

			// Allocate memory for displacements and send counts
			displs = (int *)malloc(num_of_processes * sizeof(int));
			sendcnts = (int *)malloc(num_of_processes * sizeof(int));

			displs[0] = 0;
			// Calculate data distribution
			for (int k = 0; k < num_of_processes; k++) {
				// This correctly distributes the work among all of the processes 
				// First few may get extra work if (N % num_of_proc != 0)
				sendcnts[k] = ceil((Ns[i] - k) / (double)num_of_processes);

				// Displacements are base of the sendcount distribution
				if (k > 0)
					displs[k] = sendcnts[k - 1] + displs[k - 1];
			}

			// Calculate the current process local array size
			int recvCnt = ceil((Ns[i] - my_id) / (double)num_of_processes);
			// Reserve space for the data that is to be received
			localX = (float *)malloc(recvCnt * sizeof(float));
			localK = (float *)malloc(recvCnt * sizeof(float));
			localY = (float *)malloc(recvCnt * sizeof(float));

			// All of the processes should wait here
			MPI_Barrier(MPI_COMM_WORLD);

			// Start stopwatch
			if (my_id == 0) {
				t1 = MPI_Wtime();
			}

			// First distribution phase to calculate coefficients
			MPI_Scatterv(data.x, sendcnts, displs, MPI_FLOAT, localX, recvCnt, MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Scatterv(data.y, sendcnts, displs, MPI_FLOAT, localY, recvCnt, MPI_FLOAT, 0, MPI_COMM_WORLD);


			// Calculate coefficents
			localK[0] = localY[0] / localX[0];

			for (int k = 1; k < recvCnt; k++) {
				localK[k] = fmax(localK[k - 1], localY[k] / localX[k]);
			}

			float global_max = 0;

			// Process number one only distributes its max element to all other processes
			if (num_of_processes > 1) {

				if (my_id == 0) {
					// Send local max to every higher ranked process
					for (int k = 1; k < num_of_processes; k++) {
						MPI_Send(&localK[recvCnt - 1], 1, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
					}
				}
				else {
					// Receive data from preceeding process
					for (int k = 0; k < my_id; k++) {
						float tmp = 0;
						MPI_Recv(&tmp, 1, MPI_FLOAT, k, tag, MPI_COMM_WORLD, &status);
						global_max = fmax(global_max, tmp);
					}
					// Forward local max to all of the next processes
					for (int k = my_id + 1; k < num_of_processes; k++) {
						MPI_Send(&localK[recvCnt - 1], 1, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
					}

					// Propagate preceeding max while you can
					for (int k = 0; k < recvCnt; k++) {
						if (localK[k] < global_max)
							localK[k] = global_max;
						else
							break;
					}
				}
			}

			// Calculate visibile height
			localX[0] = fmax(localY[0] - (global_max * localX[0]), 0.0f);
			for (int k = 1; k < recvCnt; k++) {
				localX[k] = fmax(localY[k] - (localK[k - 1] * localX[k]), 0.0f);
			}

			// Gather the coefficients
			MPI_Gatherv(localX, recvCnt, MPI_FLOAT, results, sendcnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

			// No need for barrier here (Gatherv does the synchronizing)
			// Stop stopwatch
			if (my_id == 0) {
				t2 = MPI_Wtime();
				stopwatch.push_back((t2 - t1));
			}

			// Free alocated temp memorry
			free(localX);
			free(localY);
			free(localK);
		}

		// First thread should also free alocated memory
		if (my_id == 0) {
			free(data.x);
			data.x = 0;
			free(data.y);
			data.y = 0;
		}

		if (my_id == 0) {
			// Mean
			double mean = 0;
			for (int j = 0; j < NUM_ITER; j++) {
				mean += stopwatch[j];
			}
			mean = mean / NUM_ITER;

			// Standard deviation
			double sDev = 0;
			for (int j = 0; j < NUM_ITER; j++) {
				sDev += pow(stopwatch[j] - mean, 2);
			}
			sDev = sqrt(sDev / NUM_ITER);

			myfile << "No. of iterations: " << NUM_ITER << endl;
			myfile << "Mean: " << mean << endl;
			myfile << "Standard deviation: " << sDev << endl << endl;
			stopwatch.clear();
		}
	}

	myfile.close();
	MPI_Finalize();

	return 0;
}
