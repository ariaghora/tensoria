@group(0) @binding(0)
var<storage, read> input_0: array<{{input_0_type}}>;

@group(0) @binding(1)
var<storage, read> input_1: array<{{input_1_type}}>;

@group(0) @binding(2)
var<storage, read_write> output_0: array<{{output_0_type}}>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var gx: u32 = gid.x;
    var gy: u32 = gid.y;
    var M: u32 = {{M}}u;
    var N: u32 = {{N}}u;
    var K: u32 = {{K}}u;

    if (gx >= N || gy >= M) { return; }

    var sum: {{output_0_type}} = {{output_0_type}}(0);
    for (var k: u32 = 0u; k < K; k += 4u) {
        sum += input_0[gy * K + k] * input_1[k * N + gx]+
               input_0[gy * K + (k + 1u)] * input_1[(k + 1u) * N + gx] +
               input_0[gy * K + (k + 2u)] * input_1[(k + 2u) * N + gx] +
               input_0[gy * K + (k + 3u)] * input_1[(k + 3u) * N + gx];
    }
    output_0[gy * N + gx] = sum;
}