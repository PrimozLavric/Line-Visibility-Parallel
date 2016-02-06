#define WORKGROUP_SIZE	(16)

__kernel void line_visibility(__global const float *X, __global const float *Y, __global float *OUT, int size)
{
	extern __local float temp[WORKGROUP_SIZE];

	// Set global thread index						
	int gid = get_global_id(0);
	int lid = get_local_id(0);

}
