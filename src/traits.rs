use std::error::Error;
use std::sync::Arc;

use crate::session::Session;
use crate::var::{TensorDataType, Variable};

pub trait TensorProps {
    fn shape(&self) -> Vec<usize>;
    fn dtype(&self) -> TensorDataType;
}

pub trait Executor {
    fn forward(&mut self, session: &Session) -> Result<(), Box<dyn Error>>;
    fn backward(&self, var: &Arc<Variable>) -> Result<(), Box<dyn Error>>;
}