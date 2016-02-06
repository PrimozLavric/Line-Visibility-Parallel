#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include "DataGenerator.h"

#define WORKGROUP_SIZE	(16)
#define MAX_SOURCE_SIZE	16384

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

	// Wait here
	scanf("%c", &ch);

	return program;
}


float* lineVisibility(lines* L, long long  N) {
	
	cl_int ret;

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

	// Compiling kernel for device 0
	cl_program program = compileKernel(&context, device_id[0], "line_visibility.cl");

	// Kernel: object preparation
	cl_kernel kernel = clCreateKernel(program, "line_visibility", &ret);


	// Delitev dela
	size_t local_item_size = WORKGROUP_SIZE;
	size_t num_groups = ((N - 1) / local_item_size + 1);
	size_t global_item_size = num_groups*local_item_size;

	// Alokacija pomnilnika na napravi
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		N*sizeof(float), L->x, &ret);
	// kontekst, na"cin, koliko, lokacija na hostu, napaka	
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		N*sizeof(float), L->y, &ret);

	// Reserve memory for result
	float *OUT = (float*)malloc(N*sizeof(float));
	memcpy(OUT, L->y, N * sizeof(int));
	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
		N*sizeof(float), OUT, &ret);

	

	scanf("%c", &ch);

	
	// program, ime "s"cepca, napaka

	// "s"cepec: argumenti
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N);
	// "s"cepec, "stevilka argumenta, velikost podatkov, kazalec na podatke

	// "s"cepec: zagon
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		&global_item_size, &local_item_size, 0, NULL, NULL);
	// vrsta, "s"cepec, dimenzionalnost, mora biti NULL, 
	// kazalec na "stevilo vseh niti, kazalec na lokalno "stevilo niti, 
	// dogodki, ki se morajo zgoditi pred klicem

	

	// Kopiranje rezultatov
	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
		N*sizeof(float), OUT, 0, NULL, NULL);
	// branje v pomnilnik iz naparave, 0 = offset
	// zadnji trije - dogodki, ki se morajo zgoditi prej


	float sum = 0;
	for (int i = 0; i < N; i++) {
		sum += OUT[i];
		cout << OUT[i] << ", ";
	}
	cout << endl;

	printf("Result: %f\n", sum);

	// "ci"s"cenje
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	free(OUT);

	return NULL;
}


int main(void)
{
	//initalize random seed
	srand(314);


	float maxAngle = 85.0;
	float p = 0.0;
	//Test your algorithm on the following cases
	//You can also try your own depending on the hardware you have on your disposal
	//long long N = (long long)32 * pow((float)10, 6); 		p = 0.01;
	//long long N = (long long)64 * pow((float)10, 6);		p = 0.006;
	//long long N = (long long)128 * pow((float)10, 6);		p = 0.004;
	//long long N = (long long)256 * pow((float)10, 6);		p = 0.002;
	//long long N = (long long)512 * pow((float)10, 6);		p = 0.001;
	//long long N = (long long)1024 * pow((float)10, 6);	p = 0.001;

	// Generate data
	//lines data = generateData(N, p, maxAngle);

	//printf("Data size: %d MB\n", N * 2 * sizeof(float) / 1024 / 1024);

	lines data;
	float tX[8] = { 1, 3, 5, 7, 9, 10, 11, 13 };
	float tY[8] = { 2, 5, 12, 20, 20, 29, 33, 40 };
	data.x = tX;
	data.y = tY;

	float* temp = lineVisibility(&data, 8);
	system("pause");

	return 0;
}