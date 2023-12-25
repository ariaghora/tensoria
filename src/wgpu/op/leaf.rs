use tera::Context;
use uuid::Uuid;

use crate::session::Session;
use crate::wgpu::op::Op;

pub struct OpLeaf {}


impl Op for OpLeaf {
    fn setup_shader_forward(&self, id: Uuid, session: &Session, params: &mut tera::Context) {
        todo!()
    }

    fn setup_shader_backward(&self, id: Uuid, session: &Session, params: &mut Context) {
        todo!()
    }

    fn workgroups(&self, id: Uuid, session: &Session) -> [u32; 3] {
        todo!()
    }
}
