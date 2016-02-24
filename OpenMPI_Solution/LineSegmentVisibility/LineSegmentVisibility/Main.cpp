#include <iostream>
#include "DataGenerator.h"
#include <chrono>
#include <mpi.h>

using namespace std;
typedef std::chrono::steady_clock Clock;


int main(int argc, char*argv[]) {
	// MPI VARS
	int my_id;				// Process id
	int num_of_processes;	// Process count
	int destination;		// Receivers rank
	int tag = 0;			// Message tag
	char message[100];		// Message
	MPI_Status status;		// Return status
	int* displs = 0;		// Displacements array
	int* sendcnts = 0;		// Number of elements that are to be sent
	float* localX = 0;		// Local process X value pointer
	float* localY = 0;		// Local process Y value pointer

	// PROBLEM VARS
	lines data;
	long long N;

	// MPI initialization
	MPI_Init(&argc, &argv);

	// Fetch current process rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	// Fetch process count
	MPI_Comm_size(MPI_COMM_WORLD, &num_of_processes);

	float maxAngle = 85.0;
	float p = 0.0;

	// Test your algorithm on the following cases
	// You can also try your own depending on the hardware you have on your disposal
	N = (long long)32 * pow((float)10, 6); 		p = 0.01;
	// long long N = (long long)64 * pow((float)10, 6);		p = 0.006;
	// long long N = (long long)128 * pow((float)10, 6);		p = 0.004;
	// long long N = (long long)256 * pow((float)10, 6);		p = 0.002;
	// long long N = (long long)512 * pow((float)10, 6);		p = 0.001;
	// long long N = (long long)1024 * pow((float)10, 6);	p = 0.001;

	N = 1024;

	// Process with rank 0 should generate the data
	if (my_id == 0) {
		// Initalize random seed
		srand(314);

		data.x = (float*)malloc(N * sizeof(float));
		data.y = (float*)malloc(N * sizeof(float));
		for (int i = 0; i < N; i++) {
			data.x[i] = 1;
			data.y[i] = i;
		}
		data.y[0] = 1024;
		// Actual data generation
		//data = generateData(N, p, maxAngle);
		printf("Data size: %d MB\n", N * 2 * sizeof(float) / 1024 / 1024);

		// Allocate memory for displacements and send counts
		displs		= (int *)malloc(num_of_processes * sizeof(int));
		sendcnts	= (int *)malloc(num_of_processes * sizeof(int));

		// Calculate data distribution
		for (int i = 0; i < num_of_processes; i++) {
			// This correctly distributes the work among all of the processes 
			// First few may get extra work if (N % num_of_proc != 0)
			sendcnts[i] = ceil((N - i) / (double)num_of_processes);

			// Displacements are base of the sendcount distribution
			if (i == 0)
				displs[i] = 0;
			else {
				displs[i] = displs[i - 1] + sendcnts[i - 1];
			}
		}
	}

	// Calculate the current process local array size
	int recvCnt = ceil((N - my_id) / (double)num_of_processes);
	// Reserve space for the data that is to be received
	localX = (float *)malloc(recvCnt * sizeof(float));
	localY = (float *)malloc(recvCnt * sizeof(float));

	// First distribution phase to calculate coefficients
	MPI_Scatterv(data.x, sendcnts, displs, MPI_FLOAT, localX, recvCnt, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(data.y, sendcnts, displs, MPI_FLOAT, localY, recvCnt, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// Calculate coefficents
	for (int i = 0; i < recvCnt; i++)
		localX[i] = localY[i] / localX[i];

	// Gather the coefficients
	MPI_Gatherv(localX, recvCnt, MPI_FLOAT, data.x, sendcnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

	float *temp = (float *)malloc(N * sizeof(float));
	// Perform the max scan
	int a = MPI_Exscan(data.x, temp, N, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

	if (my_id == 0) {
		float sum = 0;

		for (int i = 0; i < N; i++) {
			sum += temp[i];
		}
		cout << sum;
	}


	// Scatter scanned coefficients for height calculations
	//MPI_Scatterv(data.x, sendcnts, displs, MPI_FLOAT, localX, recvCnt, MPI_FLOAT, 0, MPI_COMM_WORLD);

	//Size of data in MB
	MPI_Finalize();

	system("pause");

	return 0;
}