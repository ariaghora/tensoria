use tera::Context;

use crate::gpu::array::GPUArray;

pub trait Shader {
    fn shader_path(&self) -> String;
    fn prepare(&self, operands: Vec<&GPUArray>, output: &GPUArray, params: &mut Context) -> (u32, u32, u32);
}

pub struct Add {}

impl Shader for Add {
    fn shader_path(&self) -> String { "add.wgsl".into() }

    fn prepare(&self, operands: Vec<&GPUArray>, output: &GPUArray, params: &mut Context) -> (u32, u32, u32) {
        params.insert("input_0_type", &operands[0].data_type.wgsl_type());
        params.insert("input_1_type", &operands[1].data_type.wgsl_type());
        params.insert("output_0_type", &output.data_type.wgsl_type());

        let local_size_x = 256;
        let out_shape = &output.shape;
        let num_elements = out_shape.iter().fold(1, |x, y| x * y);
        let num_workgroups_x = (num_elements + local_size_x - 1) / local_size_x;
        (num_workgroups_x as u32, 1, 1)
    }
}

pub struct MatMul {}

impl Shader for MatMul {
    fn shader_path(&self) -> String { "matmul.wgsl".into() }

    fn prepare(&self, operands: Vec<&GPUArray>, output: &GPUArray, params: &mut Context) -> (u32, u32, u32) {
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
