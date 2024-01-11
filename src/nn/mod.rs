use std::fmt::Debug;
use std::ops::Add;
use std::sync::{Arc, RwLock};

use bytemuck::Pod;
use num_integer::Roots;
use num_traits::NumCast;
use rand::distributions::{Distribution, Uniform};

use crate::error::TensoriaError;
use crate::gpu::gpu_array::GetType;
use crate::tensor::Tensor;
use crate::traits::TensoriaOps;

pub trait Module<T> {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T>;
    fn to_gpu(&self) -> Result<Self, TensoriaError>
        where
            Self: Sized;
    fn parameters(&self) -> Vec<Arc<RwLock<Tensor<T>>>>;
    fn zero_grad(&mut self);
}

pub struct Linear<T> {
    w: Arc<RwLock<Tensor<T>>>,
    b: Arc<RwLock<Tensor<T>>>,
}

impl<T> Linear<T>
    where
        T: TensoriaOps + Clone + Pod + Default + Debug,
        Vec<T>: GetType,
{
    pub fn new(in_size: usize, out_size: usize) -> Result<Self, TensoriaError> {
        let mut rng = rand::thread_rng();

        let xavier_limit =
            (6.0f32 / ((in_size + out_size) as f32).sqrt());
        let uniform = Uniform::new(-xavier_limit, xavier_limit);

        let w_val = (0..in_size * out_size)
            .map(|_| T::from(uniform.sample(&mut rng)).unwrap())
            .collect();

        let mut w = Tensor::new([in_size, out_size], w_val)?;
        let mut b = Tensor::new([out_size], vec![T::from(0.0).unwrap(); out_size])?;
        w.set_requires_grad(true);
        b.set_requires_grad(true);
        Ok(Self {
            w: Arc::new(RwLock::new(w)),
            b: Arc::new(RwLock::new(b)),
        })
    }
}

impl<T> Module<T> for Linear<T>
    where
        T: TensoriaOps + Clone + Pod + Default + Debug,
        Vec<T>: GetType,
{
    fn forward(&self, x: &Tensor<T>) -> Tensor<T> {
        let w = self.w.read().unwrap();
        let b = self.b.read().unwrap();
        x.matmul(&w).add(&b)
    }

    fn to_gpu(&self) -> Result<Self, TensoriaError> {
        Ok(Self {
            w: Arc::new(RwLock::new(self.w.read().unwrap().to_gpu()?)),
            b: Arc::new(RwLock::new(self.b.read().unwrap().to_gpu()?)),
        })
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor<T>>>> {
        vec![self.w.clone(), self.b.clone()]
    }

    fn zero_grad(&mut self) {
        self.w.write().unwrap().zero_grad();
        self.b.write().unwrap().zero_grad();
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
