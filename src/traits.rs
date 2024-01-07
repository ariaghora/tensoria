use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;

use bytemuck::Pod;
use num_traits::{NumCast, NumOps};

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

pub trait ArithmeticOps: Clone + NumCast + NumOps + Default {}

impl<T> ArithmeticOps for T where T: Clone + NumCast + NumOps + Default {}

pub trait GPUType: Clone + Pod + Default + Debug {}

impl<T> GPUType for T where
    T: Clone + Pod + Default + Debug,
{}
