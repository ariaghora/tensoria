use tera::Context;

use crate::gpu::gpu_array::{compute_broadcasted_shape_and_strides, GPUArray};

pub trait Shader {
    fn shader_path(&self) -> String;
    fn prepare<T>(
        &self,
        operands: Vec<&GPUArray<T>>,
        output: &GPUArray<T>,
        params: &mut Context,
    ) -> (u32, u32, u32);
}

macro_rules! define_elementwise_binop {
    ($struct_name: ident, $binop_stmt: expr) => {
        pub struct $struct_name {}
        impl Shader for $struct_name {
            fn shader_path(&self) -> String {
                "binop.wgsl".into()
            }
            fn prepare<T>(
                &self,
                operands: Vec<&GPUArray<T>>,
                output: &GPUArray<T>,
                params: &mut Context,
            ) -> (u32, u32, u32) {
                prepare_binop_broadcast_shader(operands, output, params, $binop_stmt)
            }
        }
    };
}

macro_rules! define_reduce_to_scalar {
    ($struct_name: ident, $reduction_stmt:expr, $postproc_stmt:expr) => {
        pub struct $struct_name {}
        impl Shader for $struct_name {
            fn shader_path(&self) -> String {
                "reduce_to_scalar.wgsl".into()
            }
            fn prepare<T>(
                &self,
                operands: Vec<&GPUArray<T>>,
                output: &GPUArray<T>,
                params: &mut Context,
            ) -> (u32, u32, u32) {
                prepare_reduction_shader(operands, output, params, $reduction_stmt, $postproc_stmt)
            }
        }
    };
}

define_elementwise_binop!(Add, "out = lhs + rhs;");
define_elementwise_binop!(Mul, "out = lhs * rhs;");
define_elementwise_binop!(Sub, "out = lhs - rhs;");
define_elementwise_binop!(Div, "out = lhs / rhs;");

define_reduce_to_scalar!(
    Mean,
    "out = lhs + rhs;",
    "out = out / output_type(input_len);"
);
define_reduce_to_scalar!(Sum, "out = lhs + rhs;", "");

pub struct MatMul {}

impl Shader for MatMul {
    fn shader_path(&self) -> String {
        "matmul.wgsl".into()
    }

    fn prepare<T>(
        &self,
        operands: Vec<&GPUArray<T>>,
        output: &GPUArray<T>,
        params: &mut Context,
    ) -> (u32, u32, u32) {
        let m = operands[0].shape[0];
        let n = operands[1].shape[1];
        let k = operands[1].shape[0];

        params.insert("input_0_type", &operands[0].data_type.wgsl_type());
        params.insert("input_1_type", &operands[1].data_type.wgsl_type());
        params.insert("output_0_type", &output.data_type.wgsl_type());
        params.insert("M", &m);
        params.insert("N", &n);
        params.insert("K", &k);

        let local_size_x_y = 16;
        let num_workgroups_x = (n + local_size_x_y - 1) / local_size_x_y;
        let num_workgroups_y = (m + local_size_x_y - 1) / local_size_x_y;
        let wg = (num_workgroups_x as u32, num_workgroups_y as u32, 1);
        return wg;
    }
}

pub struct Slice {
    slice_axis: i32,
}

impl Slice {
    pub fn new(axis: i32) -> Self {
        Slice { slice_axis: axis }
    }
}

impl Shader for Slice {
    fn shader_path(&self) -> String {
        "slice.wgsl".into()
    }

    fn prepare<T>(
        &self,
        operands: Vec<&GPUArray<T>>,
        output: &GPUArray<T>,
        params: &mut Context,
    ) -> (u32, u32, u32) {
        params.insert("input_type", &operands[0].data_type.wgsl_type());
        params.insert("indices_type", &operands[1].data_type.wgsl_type());
        params.insert("output_type", &output.data_type.wgsl_type());
        // Input shape now adjusted to be after slice, i.e., similar to that of the output
        params.insert("input_shape_csv", &vec_to_csv(&output.shape));
        params.insert("input_strides_csv", &vec_to_csv(&operands[0].strides));
        params.insert("input_ndim", &operands[0].shape.len());
        params.insert(
            "indices_len",
            &operands[1].shape.iter().fold(1, |x, y| x * y),
        );
        params.insert("output_shape_csv", &vec_to_csv(&output.shape));
        params.insert("output_strides_csv", &vec_to_csv(&output.strides));
        params.insert("output_ndim", &output.shape.len());
        params.insert("output_len", &output.shape.iter().fold(1, |x, y| x * y));
        params.insert("slicing_axis", &self.slice_axis);
        params.insert(
            "nd_index_init",
            &vec_to_csv(&vec![0; operands[0].shape.len()]),
        );

        let local_size_x = 256;
        let out_shape = &output.shape;
        let num_elements = out_shape.iter().fold(1, |x, y| x * y);
        let num_workgroups_x = (num_elements + local_size_x - 1) / local_size_x;
        (num_workgroups_x as u32, 1, 1)
    }
}

fn vec_to_csv(shape: &Vec<usize>) -> String {
    shape
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<String>>()
        .join(",")
}

fn generate_idx_code(
    idx_var_name: &str,
    shape: &Vec<usize>,
    adjusted_strides: &Vec<usize>,
) -> String {
    let mut code = String::new();
    let mut division_products = vec![1; shape.len()];

    // Precompute division products in reverse order
    for i in (0..shape.len() - 1).rev() {
        division_products[i] = division_products[i + 1] * shape[i + 1];
    }

    let mut terms = Vec::new();
    for i in 0..shape.len() {
        let term = format!(
            "((idx / {}u) % {}u) * {}u",
            division_products[i], shape[i], adjusted_strides[i]
        );
        terms.push(term);
    }

    let compiled_terms = terms.join(" + ");
    code.push_str(&format!("{} = {};\n", idx_var_name, compiled_terms));

    code
}

/// TODO: Handle specific case:
///     - tensor-scalar
///     - scalar-tensor
fn prepare_binop_broadcast_shader<T>(
    operands: Vec<&GPUArray<T>>,
    output: &GPUArray<T>,
    params: &mut Context,
    binop_stmt: &str,
) -> (u32, u32, u32) {
    params.insert("input_0_type", &operands[0].data_type.wgsl_type());
    params.insert("input_1_type", &operands[1].data_type.wgsl_type());
    params.insert("output_0_type", &output.data_type.wgsl_type());

    let (shape0, shape1) = (&operands[0].shape, &operands[1].shape);
    let (strides1, strides2) = (&operands[0].strides, &operands[1].strides);

    let (adj_shape0, adj_shape1, adj_strides0, adj_strides1) = if shape0 == shape1 {
        (
            shape0.clone(),
            shape1.clone(),
            strides1.clone(),
            strides2.clone(),
        )
    } else {
        compute_broadcasted_shape_and_strides(&shape0, &shape1, &strides1, &strides2)
    };

    let left_broadcast = shape0 != &adj_shape0;
    let right_broadcast = shape1 != &adj_shape1;
    if left_broadcast {
        params.insert("left_broadcast", &true);
        let left_numel = shape1.iter().fold(1, |x, y| x * y);
        // if lhs element length is 1 (scalar), then don't generate anything
        let idx0_code = if left_numel == 0 {
            "".into()
        } else {
            generate_idx_code("idx0", &adj_shape0, &adj_strides0)
        };
        params.insert("idx0_code", &idx0_code);
    }
    if right_broadcast {
        params.insert("right_broadcast", &true);
        let right_numel = shape1.iter().fold(1, |x, y| x * y);
        // if rhs element length is 1 (scalar), then don't generate anything
        let idx1_code = if right_numel == 0 {
            "".into()
        } else {
            generate_idx_code("idx1", &adj_shape1, &adj_strides1)
        };
        params.insert("idx1_code", &idx1_code);
    }

    params.insert("input_0_shape_csv", &vec_to_csv(&adj_shape0));
    params.insert("input_0_strides_csv", &vec_to_csv(&adj_strides0));
    params.insert("input_0_ndim", &adj_shape0.len());
    params.insert("input_1_shape_csv", &vec_to_csv(&adj_shape1));
    params.insert("input_1_strides_csv", &vec_to_csv(&adj_strides1));
    params.insert("input_1_ndim", &adj_shape1.len());
    params.insert("output_shape_csv", &vec_to_csv(&output.shape));
    params.insert("output_ndim", &output.shape.len());
    params.insert("output_len", &output.shape.iter().fold(1, |x, y| x * y));

    params.insert("binop_stmt", binop_stmt);

    let local_size_x = 16; // Use a smaller workgroup size for better distribution
    let local_size_y = 16;

    let out_shape = &output.shape;
    let num_elements = out_shape.iter().fold(1, |x, y| x * y);

    // Calculate the number of workgroups needed in each dimension
    let num_workgroups_x = ((out_shape[0] + local_size_x - 1) / local_size_x) as u32;
    let num_workgroups_y = ((num_elements / out_shape[0] + local_size_y - 1) / local_size_y) as u32;

    (num_workgroups_x, num_workgroups_y, 1)
}

fn prepare_reduction_shader<T>(
    operands: Vec<&GPUArray<T>>,
    output: &GPUArray<T>,
    params: &mut Context,
    reduction_stmt: &str,
    postproc_stmt: &str,
) -> (u32, u32, u32) {
    let input_shape = &operands[0].shape;
    let input_len = input_shape.iter().fold(1, |x, y| x * y);
    params.insert("input_type", &operands[0].data_type.wgsl_type());
    params.insert("output_type", &output.data_type.wgsl_type());
    params.insert("input_len", &input_len);
    params.insert("reduction_stmt", &reduction_stmt);
    params.insert("postproc_stmt", &postproc_stmt);

    (1, 1, 1)
}
