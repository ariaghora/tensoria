use tera::Context;
use uuid::Uuid;

use crate::session::Session;
use crate::wgpu::op::Op;

pub struct OpMatmul {}

impl Op for OpMatmul {
    fn setup_shader_forward(&self, id: Uuid, session: &Session, params: &mut Context) {
        let op = &session.variables.borrow()[&id];
        let left = &session.variables.borrow()[&op.prevs[0]];
        let right = &session.variables.borrow()[&op.prevs[1]];
        params.insert("input_0_type", left.dtype.wgsl_type());
        params.insert("input_1_type", right.dtype.wgsl_type());
        params.insert("output_0_type", op.dtype.wgsl_type());
        params.insert("M", &left.shape[0]);
        params.insert("N", &right.shape[1]);
        params.insert("K", &left.shape[1]);
        params.insert("n_unroll", &1);
    }

    fn setup_shader_backward(&self, _id: Uuid, _session: &Session, _params: &mut Context) {
        todo!()
    }

    fn workgroups(&self, id: Uuid, session: &Session) -> [u32; 3] {
        let local_size_x_y = 16;

        let out_shape = &session.variables.borrow()[&id].shape;
        let m = out_shape[0];
        let n = out_shape[1];
        let num_workgroups_x = (n + local_size_x_y - 1) / local_size_x_y;
        let num_workgroups_y = (m + local_size_x_y - 1) / local_size_x_y;
        let wg = [num_workgroups_x as u32, num_workgroups_y as u32, 1];
        return wg;
    }
}
