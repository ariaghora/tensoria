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

fn shape_to_csv(shape: &Vec<usize>) -> String {
    shape
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<String>>()
        .join(",")
}


fn generate_idx_code(idx_var_name: &str, shape: &Vec<usize>, adjusted_strides: &Vec<usize>) -> String {
    let mut code = String::new();
    let mut terms = Vec::new();
    for i in 0..shape.len() {
        let mut division_product = 1;
        for j in i + 1..shape.len() {
            division_product *= shape[j];
        }
        let term = format!("((idx / {}u) % {}u) * {}u", division_product, shape[i], adjusted_strides[i]);
        terms.push(term);
    }
    let compiled_terms = terms.join("+");
    code.push_str(&format!("{} += {};\n", idx_var_name, compiled_terms));

    code
}

fn prepare_binop_broadcast_shader<T>(operands: Vec<&GPUArray<T>>, output: &GPUArray<T>, params: &mut Context, binop_stmt: &str) -> (u32, u32, u32) {
    params.insert("input_0_type", &operands[0].data_type.wgsl_type());
    params.insert("input_1_type", &operands[1].data_type.wgsl_type());
    params.insert("output_0_type", &output.data_type.wgsl_type());

    let (shape0, shape1) = (&operands[0].shape, &operands[1].shape);
    let (strides1, strides2) = (&operands[0].strides, &operands[1].strides, );

    let (adj_shape0, adj_shape1, adj_strides0, adj_strides1) = if shape0 == shape1 {
        (shape0.clone(), shape1.clone(), strides1.clone(), strides2.clone())
    } else {
        compute_broadcasted_shape_and_strides(&shape0, &shape1, &strides1, &strides2)
    };

    let left_broadcast = shape0 != &adj_shape0;
    let right_broadcast = shape1 != &adj_shape1;
    if left_broadcast {
        let idx0_code = generate_idx_code("idx0", &adj_shape0, &adj_strides0);
        params.insert("left_broadcast", &true);
        params.insert("idx0_code", &idx0_code);
    }
    if right_broadcast {
        let idx1_code = generate_idx_code("idx1", &adj_shape1, &adj_strides1);
        params.insert("right_broadcast", &true);
        params.insert("idx1_code", &idx1_code);
    }

    params.insert("input_0_shape_csv", &shape_to_csv(&adj_shape0));
    params.insert("input_0_strides_csv", &shape_to_csv(&adj_strides0));
    params.insert("input_0_ndim", &adj_shape0.len());
    params.insert("input_1_shape_csv", &shape_to_csv(&adj_shape1));
    params.insert("input_1_strides_csv", &shape_to_csv(&adj_strides1));
    params.insert("input_1_ndim", &adj_shape1.len());
    params.insert("output_shape_csv", &shape_to_csv(&output.shape));
    params.insert("output_ndim", &output.shape.len());
    params.insert("output_len", &output.shape.iter().fold(1, |x, y| x * y));

    params.insert("binop_stmt", binop_stmt);

    let local_size_x = 256;
    let out_shape = &output.shape;
    let num_elements = out_shape.iter().fold(1, |x, y| x * y);
    let num_workgroups_x = (num_elements + local_size_x - 1) / local_size_x;
    (num_workgroups_x as u32, 1, 1)
}

pub struct Add {}

impl Shader for Add {
    fn shader_path(&self) -> String {
        "binop.wgsl".into()
    }

    fn prepare<T>(&self, operands: Vec<&GPUArray<T>>, output: &GPUArray<T>, params: &mut Context) -> (u32, u32, u32) {
        prepare_binop_broadcast_shader(operands, output, params, "out = lhs + rhs;")
    }
}

pub struct Mul {}

impl Shader for Mul {
    fn shader_path(&self) -> String { "binop.wgsl".into() }

    fn prepare<T>(&self, operands: Vec<&GPUArray<T>>, output: &GPUArray<T>, params: &mut Context) -> (u32, u32, u32) {
        prepare_binop_broadcast_shader(operands, output, params, "out = lhs * rhs;")
    }
}

pub struct Sub {}

impl Shader for Sub {
    fn shader_path(&self) -> String { "binop.wgsl".into() }

    fn prepare<T>(&self, operands: Vec<&GPUArray<T>>, output: &GPUArray<T>, params: &mut Context) -> (u32, u32, u32) {
        prepare_binop_broadcast_shader(operands, output, params, "out = lhs - rhs;")
    }
}

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
        let out_shape = &output.shape;
        let m = out_shape[0];
        let n = out_shape[1];
        let k = out_shape[0];

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
