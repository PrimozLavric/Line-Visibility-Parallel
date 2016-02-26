#include <iostream>
#include <mpi.h>
#include <string.h> 
#include <stdio.h>
#include <iostream>
#include <time.h> 
#include "DataGenerator.h"
#include <algorithm>

using namespace std;

int main(int argc, char*argv[]) {
	// MPI VARS
	int my_id;				// Process id
	int num_of_processes;	// Process count
	int destination;		// Receivers rank
	MPI_Status status;		// Return status
	int* displs = 0;		// Displacements array
	int* sendcnts = 0;		// Number of elements that are to be sent
	float* localX = 0;		// Local process X value pointer
	float* localY = 0;		// Local process Y value pointer
	int tag = 0;

	// PROBLEM VARS
	lines data;
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
			data.x[i] = i + 1;
			data.y[i] = i + 1;
		}
		
		// Actual data generation
		//data = generateData(N, p, maxAngle);
		printf("Data size: %d MB\n", N * 2 * sizeof(float) / 1024 / 1024);
	}

	// Allocate memory for displacements and send counts
	displs		= (int *)malloc(num_of_processes * sizeof(int));
	sendcnts	= (int *)malloc(num_of_processes * sizeof(int));

	// Calculate data distribution
	for (int i = 0; i < num_of_processes; i++) {
		// This correctly distributes the work among all of the processes 
		// First few may get extra work if (N % num_of_proc != 0)
		sendcnts[i] = ceil((N - i) / (double)num_of_processes);

		// Displacements are base of the sendcount distribution
		displs[i] = 0;
	}
	


	// Calculate the current process local array size
	int recvCnt = ceil((N - my_id) / (double)num_of_processes);
	// Reserve space for the data that is to be received
	localX = (float *)malloc(recvCnt * sizeof(float));
	localY = (float *)malloc(recvCnt * sizeof(float));


	// First distribution phase to calculate coefficients
	MPI_Scatterv(data.x, sendcnts, displs, MPI_FLOAT, localX, recvCnt, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(data.y, sendcnts, displs, MPI_FLOAT, localY, recvCnt, MPI_FLOAT, 0, MPI_COMM_WORLD);

	cout << "ID: " << my_id << " recvCnt: " << recvCnt << endl;

	// Calculate coefficents
	localX[0] = localY[0] / localX[0];
	for (int i = 0; i < recvCnt; i++) {
		localX[i] = max(localX[i-1], localY[i] / localX[i]);
	}

	// Process number one only distributes its max element to all other processes
	if (num_of_processes > 1) {
		if (my_id == 0) {
			for (int i = 1; i < num_of_processes; i++) {
				MPI_Send(&localX[recvCnt - 1], 1, MPI_FLOAT, i, tag, MPI_COMM_WORLD);
			}
		}
		else {
			int global_max = 0;
			// Receive data from preceeding process
			for (int i = 0; i < my_id; i++) {
				int tmp = 0;
				MPI_Recv(&tmp, 1, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
				global_max = max(global_max, tmp);
			}
			// Forward local max to all of the next processes
			for (int i = my_id + 1; i < num_of_processes; i++) {
				MPI_Send(&localX[recvCnt - 1], 1, MPI_FLOAT, i, tag, MPI_COMM_WORLD);
			}

			for (int i = 0; i > recvCnt; i++) {
				if (localX[i] < global_max)
					localX[i] = global_max;
				else
					break;
			}
		}
	}

	float sum = 0;

	for (int i = 0; i < recvCnt; i++) {
		sum += localX[i];
	}
	cout << "ID: " << my_id << " Sum: " << sum << endl;

	// Gather the coefficients
	MPI_Gatherv(localX, recvCnt, MPI_FLOAT, data.x, sendcnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);


	if (my_id == 0) {
		float sum = 0;

		for (int i = 0; i < N; i++) {
			sum += data.x[i];
		}
		cout << sum << endl;
	}


	// Scatter scanned coefficients for height calculations
	//Size of data in MB
	MPI_Finalize();

	if (my_id == 0)
		system("pause");

	return 0;
}
