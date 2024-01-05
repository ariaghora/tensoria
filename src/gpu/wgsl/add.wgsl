@group(0) @binding(0)
var<storage, read_write> input_0: array<{{input_0_type}}>;

@group(0) @binding(1)
var<storage, read_write> input_1: array<{{input_1_type}}>;

@group(0) @binding(2)
var<storage, read_write> output_0: array<{{output_0_type}}>;

const input_0_shape   = array<i32, {{input_0_ndim}}>( {{input_0_shape_csv}} );
const input_0_strides = array<i32, {{input_0_ndim}}>( {{input_0_strides_csv}} );
const input_1_shape   = array<i32, {{input_1_ndim}}>( {{input_1_shape_csv}} );
const input_1_strides = array<i32, {{input_1_ndim}}>( {{input_1_strides_csv}} );
const output_shape    = array<i32, {{output_ndim}}>( {{output_shape_csv}} );

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var idx: u32 = gid.x;

    if idx >= {{output_len}}u { return; }

    {% if left_broadcast -%}
    var idx0 = 0u;
    {{idx0_code}}
    {% else -%}
    var idx0 = idx;
    {% endif %}

    {%- if right_broadcast -%}
    var idx1 = 0u;
    {{idx1_code}}
    {%- else -%}
    var idx1 = idx;
    {% endif %}

    output_0[idx] = input_0[idx0] + input_1[idx1];
}