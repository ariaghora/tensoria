use tera::Context;
use uuid::Uuid;

use crate::session::Session;
use crate::wgpu::op::Op;

pub struct OpLeaf {}

impl Op for OpLeaf {
    fn setup_shader_forward(&self, _id: Uuid, _session: &Session, _params: &mut tera::Context) {
        todo!()
    }

    fn setup_shader_backward(&self, _id: Uuid, _session: &Session, _params: &mut Context) {
        todo!()
    }

    fn workgroups(&self, _id: Uuid, _session: &Session) -> [u32; 3] {
        todo!()
    }
}
