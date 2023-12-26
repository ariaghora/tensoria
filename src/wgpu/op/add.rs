use tera::Context;
use uuid::Uuid;

use crate::session::Session;
use crate::wgpu::op::Op;

pub struct OpAdd {}

impl Op for OpAdd {
    fn setup_shader_forward(&self, id: Uuid, session: &Session, params: &mut tera::Context) {
        let op = &session.variables.borrow()[&id];
        let left = &session.variables.borrow()[&op.prevs[0]];
        let right = &session.variables.borrow()[&op.prevs[1]];
        params.insert("input_0_type", left.dtype.wgsl_type());
        params.insert("input_1_type", right.dtype.wgsl_type());
        params.insert("output_0_type", op.dtype.wgsl_type());
    }

    fn setup_shader_backward(&self, _id: Uuid, _session: &Session, _params: &mut Context) {
        todo!()
    }

    fn workgroups(&self, id: Uuid, session: &Session) -> [u32; 3] {
        let local_size_x = 256;

        let out_shape = &session.variables.borrow()[&id].shape;
        let num_elements = out_shape.iter().fold(1, |x, y| x * y);
        let num_workgroups_x = (num_elements + local_size_x - 1) / local_size_x;
        [num_workgroups_x as u32, 1, 1]
    }
}
