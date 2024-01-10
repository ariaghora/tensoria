@group(0) @binding(0)
var<storage, read_write> input: array<{{input_type}}>;

@group(0) @binding(1)
var<storage, read_write> indices: array<{{indices_type}}>;

@group(0) @binding(2)
var<storage, read_write> output: array<{{output_type}}>;


@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var input_shape     = array<i32, {{input_ndim}}>( {{input_shape_csv}} );
    var input_strides   = array<i32, {{input_ndim}}>( {{input_strides_csv}} );
    var output_shape    = array<i32, {{output_ndim}}>( {{output_shape_csv}} );
    var output_strides  = array<i32, {{output_ndim}}>( {{output_strides_csv}} );
    var slicing_axis    = {{slicing_axis}};

    var offset = i32(gid.x);
    if (offset >= {{output_len}}) { return; }

    var nd_index = array<i32, {{input_ndim}}>( {{nd_index_init}} );
    for (var i = {{input_ndim}} - 1; i >= 0; i--) {
        if (i == slicing_axis) {
            nd_index[i] = indices[offset % input_shape[i]];
        } else {
            nd_index[i] = offset % input_shape[i];
        }
        offset /= input_shape[i];
    }

    var logical_offset = 0;
    for (var i = 0; i < {{input_ndim}}; i++) {
        logical_offset += nd_index[i] * input_strides[i];
    }

    output[gid.x] = input[logical_offset];
}