use uuid::Uuid;

use crate::session::Session;

pub mod add;
pub mod matmul;
pub mod leaf;
pub mod mul;

pub trait Op {
    fn setup_shader_forward(&self, id: Uuid, session: &Session, params: &mut tera::Context);
    fn setup_shader_backward(&self, id: Uuid, session: &Session, params: &mut tera::Context);
    fn workgroups(&self, id: Uuid, session: &Session) -> [u32; 3];
}

