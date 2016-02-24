#define WORKGROUP_SIZE 1024
#define BATCH_SIZE (2 * WORKGROUP_SIZE)

__kernel void propagate_maximums(__global float *propagated, __global float *maximums)
{
	// Retrive indices
	int gid = get_global_id(0);
	int wid = get_group_id(0);

	// Propagate backwards
	propagated[2 * gid] = fmax(maximums[wid], propagated[2 * gid]);
	propagated[2 * gid + 1] = fmax(maximums[wid], propagated[2 * gid + 1]);
}
