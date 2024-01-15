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
    use crate::{error::TensoriaError, tensor::Tensor};

    use super::softmax_unstable;

    #[test]
    fn softmax() -> Result<(), TensoriaError> {
        let x = Tensor::new([3, 2], vec![2., 2., 4., 4., 6., 6.])?;
        let res = softmax_unstable(&x, 1);
        assert_eq!(res.sum(Some(1), false).to_vec(), vec![1., 1., 1.]);
        Ok(())
    }
}
