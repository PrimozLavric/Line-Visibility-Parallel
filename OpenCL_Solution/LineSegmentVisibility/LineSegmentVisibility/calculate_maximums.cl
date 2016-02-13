#define WORKGROUP_SIZE 1024
#define BATCH_SIZE (2 * WORKGROUP_SIZE)

__kernel void calculate_maximums(__global float *input, __global float *maximums, int group_count)
{
	__local float local_mem[WORKGROUP_SIZE * 2];

	// Retrive indices
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int wid = get_group_id(0);

	// Copy to local memory and calculate first iteration in the process
	local_mem[2 * lid] = input[2 * gid];
	local_mem[2 * lid + 1] = max(local_mem[2 * lid], input[2 * gid + 1]);

	// UPSWEEP
	// Start at step 2 (Step one is already calculated)
	int offset = 2;
	for (int i = BATCH_SIZE >> 2; i > 0; i >>= 1) {
		// Synchronizing every iteration
		barrier(CLK_LOCAL_MEM_FENCE);

		// Check if thread has to do anything
		if (lid < i) {
			int ai = offset * (2 * lid + 1) - 1;
			int bi = offset * (2 * lid + 2) - 1;

			local_mem[bi] = max(local_mem[ai], local_mem[bi]);
		}

		// Increment step size
		offset <<= 1;
	}

	// Temporaly copy last element from coefficient array and set it to 0
	if (lid == 0) {
		// Store maximal value of this group if this workgroup had not covered the whole array
		if (!group_count <= 1)
			maximums[wid] = local_mem[BATCH_SIZE - 1];
		local_mem[BATCH_SIZE - 1] = 0;
	}

	// DOWNSWEEP
	for (int i = 1; i < BATCH_SIZE; i <<= 1) {
		// Decrement step size
		offset >>= 1;

		//Synchronizing every iteration
		barrier(CLK_LOCAL_MEM_FENCE);

		// Check if thread has to do anything
		if (lid < i) {
			int ai = offset*(2 * lid + 1) - 1;
			int bi = offset*(2 * lid + 2) - 1;

			float tmp = local_mem[ai];
			local_mem[ai] = local_mem[bi];
			local_mem[bi] = max(local_mem[bi], tmp);
		}
	}

	// Wait for everybody to finish
	barrier(CLK_LOCAL_MEM_FENCE);

	// Write calculated coefficients back to global memory
	input[2 * gid] = local_mem[2 * lid];
	input[2 * gid + 1] = local_mem[2 * lid + 1];
}
