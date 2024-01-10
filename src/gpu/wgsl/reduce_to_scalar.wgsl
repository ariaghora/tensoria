alias input_type = {{input_type}};
alias output_type = {{output_type}};

@group(0) @binding(0)
var<storage, read> input: array<input_type>;

@group(0) @binding(1)
var<storage, read_write> output: array<output_type>;

// Local memory for partial sums
var<workgroup> sums: array<output_type, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    var local_sum: input_type = input_type(0);
    let group_size = 256u;
    let input_len = {{input_len}}u;
    let num_groups = (input_len + group_size - 1u) / group_size;
    let group_id = global_id.x / group_size;
    let local_id = global_id.x % group_size;

    // Each thread in the workgroup adds up its subset of the input
    for (var i = local_id; i < input_len; i += group_size) {
        let lhs = local_sum;
        let rhs = input[i];
        let out = lhs + rhs;
        local_sum = out;
    }

    // Store the sum in shared memory
    sums[local_id] = local_sum;
    workgroupBarrier();

    // Reduction in shared memory
    for (var stride = group_size / 2u; stride > 0u; stride /= 2u) {
        if (local_id < stride) {
            let lhs = sums[local_id];
            let rhs = sums[local_id + stride];
            var out: output_type;
            {{reduction_stmt}};
            sums[local_id] = out;
        }
        workgroupBarrier();
    }

    // Write the result to the output array
    if (local_id == 0u) {
        var out: output_type;
        out = sums[0];
        {{postproc_stmt}};
        output[group_id] = out;
    }
}
