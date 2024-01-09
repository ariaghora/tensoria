use std::fmt::Debug;

use bytemuck::Pod;
use num_traits::{FromPrimitive, Num, NumCast, NumOps, Zero};

pub trait ArithmeticOps: Clone + Num + NumCast + NumOps + PartialOrd + Default + Zero + FromPrimitive {}

impl<T> ArithmeticOps for T where T: Clone + Num + NumCast + NumOps + PartialOrd + Default + Zero + FromPrimitive {}

pub trait GPUType: Clone + Pod + Default + Debug {}

impl<T> GPUType for T where
    T: Clone + Pod + Default + Debug,
{}
