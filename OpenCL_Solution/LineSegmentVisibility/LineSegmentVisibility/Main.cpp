#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include "DataGenerator.h"
#include <vector>
#include <chrono>
#include <algorithm>

#define WORKGROUP_SIZE 1024
#define BATCH_SIZE (2 * WORKGROUP_SIZE)
#define MAX_SOURCE_SIZE	16384

typedef std::chrono::steady_clock Clock;

using namespace std;
char ch;

cl_program compileKernel(cl_context *context, cl_device_id deviceID, char* kernelPath) {
	// cl_int for storing returning status code
	cl_int ret;

	// Reads kernel from file
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen(kernelPath, "r");
	if (!fp)
	{
		fprintf(stderr, ":-(#\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
	fclose(fp);

	// Creating program from source
	// Context, number of source pointers, source pointers, strings are NULL terminated, output code
	cl_program program = clCreateProgramWithSource(*context, 1, (const char **)&source_str, NULL, &ret);
												
	// Building from cource code
	// Program, number of devices, id of device to bild for, build options, pointer to function, user arguments
	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

	// Generating buildlog for debugging
	size_t build_log_len;
	char *build_log;

	// Program, deviceID, type of log, max string length, pointer to string, log length
	// In first itereation fetch the size of the log
	ret = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);

	// Reserve space for build log
	build_log = (char *)malloc(sizeof(char)*(build_log_len + 1));
	// Write log to reserved string
	ret = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
	// Print build log
	printf("Build Log:\n%s", build_log);
	free(build_log);

	return program;
}

// Helper function to calculate logarithm with base n.
// Returns ceiled logarithm
inline int logBase(float x, float n) {
	return ceil(log(x) / log(n));
}


float* lineVisibility(lines* L, long long  N) {

	cl_int ret;

	// Vectors for storing opencl data for easyer deallocation
	vector<cl_kernel> kernels;
	vector<cl_program> programs;

	// \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
		// ************** OpenCL INITIALIZATION **************
	// ///////////////////////////////////////////////////
	// Read platform data
	cl_platform_id	platform_id[10];
	cl_uint			ret_num_platforms;
	char			*buf;
	size_t			buf_len;
	// Max number of platforms, platform array to store in, actual platform count
	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);

	// Fetch device data
	cl_device_id	device_id[10];
	cl_uint			ret_num_devices;
	// We will work with platform[0] (GPU)
	// Selected platform, device type, max number of devices, pointer to devices, actual device count
	ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);

	// Generating OpenCL context
	// Context propreties: NULL default, device cout, devices ids, 
	// pointer to call-back function, extra function parameters, status number
	cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);

	// Creating comand queue
	// Context, device, INORDER/OUTOFORDER, status code
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);

	// Kernel 1
	programs.push_back(compileKernel(&context, device_id[0], "line_visibility.cl"));
	kernels.push_back(clCreateKernel(programs[0], "line_visibility", &ret));
	// Kernel 2
	programs.push_back(compileKernel(&context, device_id[0], "calculate_maximums.cl"));

	programs.push_back(compileKernel(&context, device_id[0], "propagate_maximums.cl"));

	programs.push_back(compileKernel(&context, device_id[0], "propagation_height_calc.cl"));
	// Wait here
	scanf("%c", &ch);
	auto t1 = Clock::now();
	// \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
		// **************** WORK DISTRIBUTION ****************
	// ///////////////////////////////////////////////////

	// TODO :: It is expected that N is divisibile with WORKGROUP_SIZE * 2
	size_t local_item_size = WORKGROUP_SIZE;

	// X and Y presenting distance from camera and height of linse
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		N*sizeof(float), L->x, &ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		N*sizeof(float), L->y, &ret);
	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
		N*sizeof(float), NULL, &ret);

	// Calculating subiterations sizes
	//int subinterCount = logBase(group_count, BATCH_SIZE);

	// Calculating and alocating sub iteration data
	vector<size_t> group_counts;
	vector<size_t> global_item_sizes;
	vector<cl_mem> kernel_data;

	// Main iteration
	group_counts.push_back(N / BATCH_SIZE);
	global_item_sizes.push_back(group_counts[0] * local_item_size);
	kernel_data.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, (group_counts[0] + BATCH_SIZE - (group_counts[0] % BATCH_SIZE))  * sizeof(float), NULL, &ret));

	// Sub iterations
	int i = 1;
	do {
		group_counts.push_back(ceil((float)group_counts[i-1] / BATCH_SIZE));
		global_item_sizes.push_back(group_counts[i] * local_item_size);
		int a = (group_counts[i] + BATCH_SIZE - (group_counts[i] % BATCH_SIZE));
		kernel_data.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, (group_counts[i] + BATCH_SIZE - (group_counts[i] % BATCH_SIZE))  * sizeof(float), NULL, &ret));
	} while (group_counts[i++] != 1);
	
	// Defining first kernel arguments
	ret = clSetKernelArg(kernels[0], 0, sizeof(cl_mem), (void *)&a_mem_obj);
	ret |= clSetKernelArg(kernels[0], 1, sizeof(cl_mem), (void *)&b_mem_obj);
	ret |= clSetKernelArg(kernels[0], 2, sizeof(cl_mem), (void *)&c_mem_obj);
	ret |= clSetKernelArg(kernels[0], 3, sizeof(cl_mem), (void *)&kernel_data[0]);
	
	// Add main line visibility kernel to queue
	ret = clEnqueueNDRangeKernel(command_queue, kernels[0], 1, NULL,
		&global_item_sizes[0], &local_item_size, 0, NULL, NULL);
	
	// Add maximum calculating kernels (upsweep)
	for (int i = 1; i < group_counts.size(); i++) {
		// Create kernel
		kernels.push_back(clCreateKernel(programs[1], "calculate_maximums", &ret));
		// Defining arguments
		ret = clSetKernelArg(kernels.back(), 0, sizeof(cl_mem), (void *)&kernel_data[i - 1]);
		ret |= clSetKernelArg(kernels.back(), 1, sizeof(cl_mem), (void *)&kernel_data[i]);
		ret |= clSetKernelArg(kernels.back(), 2, sizeof(cl_int), (void *)&group_counts[i]);

		// Enqueue kernel
		ret = clEnqueueNDRangeKernel(command_queue, kernels.back(), 1, NULL,
			&global_item_sizes[i], &local_item_size, 0, NULL, NULL);
	}

	// Downsweep
	for (int i = group_counts.size() - 2; i >= 1; i--) {
		kernels.push_back(clCreateKernel(programs[2], "propagate_maximums", &ret));

		// Pass correct memory element
		ret = clSetKernelArg(kernels.back(), 0, sizeof(cl_mem), (void *)&kernel_data[i - 1]);
		ret |= clSetKernelArg(kernels.back(), 1, sizeof(cl_mem), (void *)&kernel_data[i]);

		// Enqueue kernel
		ret = clEnqueueNDRangeKernel(command_queue, kernels.back(), 1, NULL,
			&global_item_sizes[i], &local_item_size, 0, NULL, NULL);
	}
	kernels.push_back(clCreateKernel(programs[2], "propagate_maximums", &ret));
	ret = clSetKernelArg(kernels.back(), 0, sizeof(cl_mem), (void *)&c_mem_obj);
	ret |= clSetKernelArg(kernels.back(), 1, sizeof(cl_mem), (void *)&kernel_data[0]);
	
	// Final downsweep iteration
	kernels.push_back(clCreateKernel(programs[3], "propagation_height_calc", &ret));

	// Pass correct memory element
	ret = clSetKernelArg(kernels.back(), 0, sizeof(cl_mem), (void *)&c_mem_obj);
	ret |= clSetKernelArg(kernels.back(), 1, sizeof(cl_mem), (void *)&kernel_data[0]);
	ret |= clSetKernelArg(kernels.back(), 2, sizeof(cl_mem), (void *)&b_mem_obj);
	ret |= clSetKernelArg(kernels.back(), 3, sizeof(cl_mem), (void *)&a_mem_obj);
	
	// Enqueue final kernel
	ret = clEnqueueNDRangeKernel(command_queue, kernels.back(), 1, NULL,
		&global_item_sizes[0], &local_item_size, 0, NULL, NULL);
	
	/*ret = clEnqueueReadBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
		N * sizeof(float), L->x, 0, NULL, NULL);*/
	
	
	float *T = (float*)malloc(N * sizeof(float));
	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
		N * sizeof(float), L->x, 0, NULL, NULL);
	/*
	for (int i = 0; i < N; i++) {
		L->x[i] = max(L->y[i] - T[i] * L->x[i], 0.0f);
	}*/

	auto t2 = Clock::now();
	std::cout << "Time required: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0 << "s" << endl;

	// Clear
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	for (int i = 0; i < kernels.size(); i++) {
		ret = clReleaseKernel(kernels[i]);
	}
	kernels.clear();
	for (int i = 0; i < programs.size(); i++) {
		ret = clReleaseProgram(programs[i]);
	}
	kernels.clear();
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	for (int i = 0; i < kernel_data.size(); i++) {
		ret = clReleaseMemObject(kernel_data[i]);
	}
	kernel_data.clear();
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	return L->x;
}


int main(void)
{
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

	//Size of data in MB
	printf("Data size: %d MB\n", N * 2 * sizeof(float) / 1024 / 1024);

	// Generate data
	lines data = generateData(N, p, maxAngle);

	float* temp = lineVisibility(&data, N);

	float sum = 0;
	for (int i = 0; i < N; i++) {
		sum += temp[i];
	}

	cout << sum << endl;

	system("pause");

	return 0;
}