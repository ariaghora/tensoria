@group(0) @binding(0)
var<storage, read_write> input_0: array <f32>;

@group(0) @binding(1)
var<storage, read_write> input_1: array <f32>;

@group(0) @binding(2)
var<storage, read_write> output_0: array <f32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var idx: u32 = gid.x;
    output_0[idx] = input_0[idx] + input_1[idx];
}