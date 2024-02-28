use std::fmt::Debug;

use bytemuck::Pod;
use num_traits::{FromPrimitive, Num, NumCast, NumOps, Zero};

use crate::gpu::gpu_array::GetType;

pub trait TensoriaOps:
    Sized
    + Clone
    + Num
    + NumCast
    + NumOps
    + PartialOrd
    + Default
    + Zero
    + FromPrimitive
    + Clone
    + Pod
    + Default
    + Debug
{
}

impl<T> TensoriaOps for T
where
    T: Sized
        + Clone
        + Num
        + NumCast
        + NumOps
        + PartialOrd
        + Default
        + Zero
        + FromPrimitive
        + Clone
        + Pod
        + Default
        + Debug,
    Vec<T>: GetType,
{
}

pub trait GPUType: Clone + Pod + Default + Debug {}

impl<T> GPUType for T
where
    T: Clone + Pod + Default + Debug,
    Vec<T>: GetType,
{
}
