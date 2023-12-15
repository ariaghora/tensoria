use std::error::Error;

use crate::session::Session;
use crate::var::TensorDataType;

pub trait TensorProps {
    fn shape(&self) -> Vec<usize>;
    fn dtype(&self) -> TensorDataType;
}

pub trait Executor {
    fn execute(&mut self, session: &Session) -> Result<(), Box<dyn Error>>;
}