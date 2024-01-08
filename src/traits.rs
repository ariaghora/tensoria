use std::fmt::Debug;

use bytemuck::Pod;
use num_traits::{NumCast, NumOps, Zero};

pub trait ArithmeticOps: Clone + NumCast + NumOps + Default + Zero {}

impl<T> ArithmeticOps for T where T: Clone + NumCast + NumOps + Default + Zero {}

pub trait GPUType: Clone + Pod + Default + Debug {}

impl<T> GPUType for T where
    T: Clone + Pod + Default + Debug,
{}
