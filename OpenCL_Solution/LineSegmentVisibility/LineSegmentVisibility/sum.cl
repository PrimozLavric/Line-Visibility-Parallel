__kernel void sum(__global const float *X, __global float *OUT, int size)
{
	__shared 

	// Global index of thread						
	int i = get_global_id(0);
	
	// Calculate										
	while (i < size)
	{
		OUT[i] = 1;
		i += get_global_size(0);
	}
}
