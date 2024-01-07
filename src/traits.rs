use std::fmt::Debug;

use bytemuck::Pod;
use num_traits::{NumCast, NumOps};

pub trait ArithmeticOps: Clone + NumCast + NumOps + Default {}

impl<T> ArithmeticOps for T where T: Clone + NumCast + NumOps + Default {}

pub trait GPUType: Clone + Pod + Default + Debug {}

impl<T> GPUType for T where
    T: Clone + Pod + Default + Debug,
{}
