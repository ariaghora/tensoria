use crate::gpu::gpu_array::GetType;
use crate::tensor::{GradFn, Tensor, UnOpFn};
use crate::traits::TensoriaOps;

pub fn softmax_unstable<EType: TensoriaOps>(x: &Tensor<EType>, axis: usize) -> Tensor<EType>
where
    Vec<EType>: GetType,
{
    let gf: Option<GradFn<EType>> = Some(|_lg, _og, _parent| {
        // TODO
        todo!();
    });

    let unop_fn: UnOpFn<EType> = Box::new(move |arr| {
        let nominator = arr.exp();
        &nominator / &nominator.sum(Some(axis), true)
    });
    x.tensor_unop(unop_fn, gf)
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
