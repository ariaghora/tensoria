use crate::gpu::gpu_array::GetType;
use crate::tensor::Tensor;
use crate::traits::TensoriaOps;

pub fn softmax_unstable<EType: TensoriaOps>(x: &Tensor<EType>, axis: usize) -> Tensor<EType>
where
    Vec<EType>: GetType,
{
    let nom = x.exp();
    &nom / &nom.sum(Some(axis), true)
}

#[cfg(test)]
mod test {
    use super::softmax_unstable;
    use crate::{error::TensoriaError, tensor::Tensor};

    #[test]
    fn softmax() -> Result<(), TensoriaError> {
        let mut x = Tensor::new([3, 2], vec![2., 2., 4., 4., 6., 6.])?;
        x.set_requires_grad(true);

        let res = softmax_unstable(&x, 1);
        assert_eq!(res.to_vec(), vec![0.5; 6]);
        res.backward()?;

        assert_eq!(x.grad_vec().unwrap(), vec![0.; 6]);
        Ok(())
    }
}
