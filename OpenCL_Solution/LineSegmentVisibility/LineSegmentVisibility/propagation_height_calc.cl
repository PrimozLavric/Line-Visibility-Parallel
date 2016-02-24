#define WORKGROUP_SIZE 1024
#define BATCH_SIZE (2 * WORKGROUP_SIZE)

__kernel void propagation_height_calc(__global float *propagated, __global const float *maximums, __global const float *Y, __global float *visibile_heights)
{
	// Retrive indices
	int gid = get_global_id(0);
	int wid = get_group_id(0);

	// Propagate backwards
	// Calculate visibile height
	propagated[2 * gid] = fmax(maximums[wid], propagated[2 * gid]);
	propagated[2 * gid + 1] = fmax(maximums[wid], propagated[2 * gid + 1]);
	// Visibile heights contains X values
	visibile_heights[2 * gid] = fmax(0.0f, Y[2 * gid] - visibile_heights[2 * gid] * propagated[2 * gid]);
	visibile_heights[2 * gid + 1] = fmax(0.0f, Y[2 * gid + 1] - visibile_heights[2 * gid + 1] * propagated[2 * gid + 1]);
}