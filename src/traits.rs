use std::error::Error;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

use crate::session::Session;
use crate::var::{TensorDataType, Variable};

pub trait TensorProps {
    fn shape(&self) -> Vec<usize>;
    fn dtype(&self) -> TensorDataType;
}

pub trait Executor {
    fn forward(&mut self, session: &Session) -> Result<(), Box<dyn Error>>;
    fn backward(&mut self, var: &Arc<Variable>, session: &Session) -> Result<(), Box<dyn Error>>;
}

pub trait ArithmeticOps: Clone + Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self> {}

impl<T> ArithmeticOps for T where T: Clone + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> {}
