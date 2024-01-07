use std::error::Error;
use std::sync::Arc;

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

pub trait ArithmeticOps: Clone + NumCast + NumOps {}

impl<T> ArithmeticOps for T where T: Clone + NumCast + NumOps {}
