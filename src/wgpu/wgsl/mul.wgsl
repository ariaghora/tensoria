@group(0) @binding(0)
var<storage, read> input_0: array<{{input_0_type}}>;

@group(0) @binding(1)
var<storage, read> input_1: array<{{input_1_type}}>;

@group(0) @binding(2)
var<storage, read_write> output_0: array<{{output_0_type}}>;

const input_0_shape: array<i32,4> = array(2, 2, 2, 2);
const input_1_shape: array<i32,4> = array(2, 2, 2, 2);
const input_0_strides: array<i32,4> = array(2, 2, 2, 2);
const input_1_strides: array<i32,4> = array(2, 2, 2, 2);

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var idx: u32 = gid.x;
    output_0[idx] = input_0[idx] * input_1[idx];
}