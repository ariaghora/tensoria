@group(0) @binding(0)
var<storage, read_write> input_0: array<{{input_0_type}}>;

@group(0) @binding(1)
var<storage, read_write> input_1: array<{{input_1_type}}>;

@group(0) @binding(2)
var<storage, read_write> grad_0: array<{{grad_0_type}}>;

@group(0) @binding(3)
var<storage, read_write> grad_1: array<{{grad_1_type}}>;

@group(0) @binding(4)
var<storage, read_write> output_0_grad: array<{{output_0_grad_type}}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var idx: u32 = gid.x;

    {% if left_requires_grad %}
    grad_0[idx] = input_1[idx] * output_0_grad[idx];
    {% endif %}

    {% if right_requires_grad %}
    grad_1[idx] = input_0[idx] * output_0_grad[idx];
    {% endif %}
}