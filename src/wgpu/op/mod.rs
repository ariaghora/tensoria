use uuid::Uuid;

use crate::session::Session;

pub trait Op {
    fn setup_shader(&self, id: Uuid, session: &Session, params: &mut tera::Context);
    fn workgroups(&self, id: Uuid, session: &Session) -> [u32; 3];
}

pub struct OpAdd {}

pub struct OpLeaf {}

impl Op for OpAdd {
    fn setup_shader(&self, id: Uuid, session: &Session, params: &mut tera::Context) {
        let op = &session.tensors.borrow()[&id];
        let left = &session.tensors.borrow()[&op.prevs[0]];
        let right = &session.tensors.borrow()[&op.prevs[1]];
        params.insert("input_0_type", left.dtype.wgsl_type());
        params.insert("input_1_type", right.dtype.wgsl_type());
        params.insert("output_0_type", op.dtype.wgsl_type());
    }

    fn workgroups(&self, id: Uuid, session: &Session) -> [u32; 3] {
        let local_size_x = 256;

        let out_shape = &session.tensors.borrow()[&id].shape;
        let num_elements = out_shape.iter().fold(1, |x, y| x * y);
        let num_workgroups_x = (num_elements + local_size_x - 1) / local_size_x;
        [num_workgroups_x as u32, 1, 1]
    }
}

impl Op for OpLeaf {
    fn setup_shader(&self, id: Uuid, session: &Session, params: &mut tera::Context) {
        todo!()
    }

    fn workgroups(&self, id: Uuid, session: &Session) -> [u32; 3] {
        todo!()
    }
}
