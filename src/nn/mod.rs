use std::fmt::Debug;
use std::ops::Add;
use std::sync::Arc;

use bytemuck::Pod;
use num_traits::NumCast;

use crate::error::TensoriaError;
use crate::gpu::gpu_array::GetType;
use crate::tensor::Tensor;
use crate::traits::TensoriaOps;

pub trait Module<T> {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T>;
    fn to_gpu(&self) -> Result<Self, TensoriaError> where Self: Sized;
    fn parameters(&self) -> Vec<Arc<Tensor<T>>>;
}

pub struct Linear<T> {
    w: Arc<Tensor<T>>,
    b: Arc<Tensor<T>>,
}

impl<T> Linear<T>
    where
        T: TensoriaOps + Clone + Pod + Default + Debug,
        Vec<T>: GetType,
{
    pub fn new(in_size: usize, out_size: usize) -> Result<Self, TensoriaError> {
        let w = Tensor::new(
            [in_size, out_size],
            vec![T::from(0.0).unwrap(); in_size * out_size],
        )?;
        let b = Tensor::new([out_size], vec![T::from(0.0).unwrap(); out_size])?;
        Ok(Self { w: Arc::new(w), b: Arc::new(b) })
    }
}

impl<T> Module<T> for Linear<T>
    where
        T: TensoriaOps + Clone + Pod + Default + Debug,
        Vec<T>: GetType,
{
    fn forward(&self, x: &Tensor<T>) -> Tensor<T> {
        x.matmul(&self.w).add(&self.b)
    }

    fn to_gpu(&self) -> Result<Self, TensoriaError> {
        Ok(Self {
            w: Arc::new(self.w.to_gpu()?),
            b: Arc::new(self.b.to_gpu()?),
        })
    }

    fn parameters(&self) -> Vec<Arc<Tensor<T>>> {
        vec![self.w.clone(), self.b.clone()]
    }
}


#[cfg(test)]
mod test {
    use crate::error::TensoriaError;
    use crate::nn::{Linear, Module};
    use crate::tensor::Tensor;

    #[test]
    fn linear() -> Result<(), TensoriaError> {
        let x = Tensor::new([10, 10], vec![1.0; 100])?;
        let linear = Linear::new(10, 2)?;
        let res = linear.forward(&x);

        assert_eq!(res.shape(), vec![10, 2]);

        let linear = linear.to_gpu()?;
        let res = linear.forward(&x.to_gpu()?);
        assert_eq!(res.shape(), vec![10, 2]);

        Ok(())
    }
}