use std::error::Error;
use std::ops::Add;
use std::sync::{Arc, RwLock};

use ndarray::ArrayD;

use crate::error::TensoriaError;
use crate::gpu::array::GPUArray;
use crate::traits::ArithmeticOps;

enum Device { CPU, GPU }


#[derive(Clone, Debug)]
enum ArrayData<EType> {
    CPUArray(ArrayD<EType>),
    GPUArray(GPUArray),
}

impl<EType> ArrayData<EType> {
    pub fn new_cpu<S: AsRef<[usize]>>(shape: S, data: Vec<EType>) -> Result<ArrayData<EType>, Box<dyn Error>> {
        Ok(ArrayData::CPUArray(ArrayD::from_shape_vec(shape.as_ref(), data)?))
    }
}

impl<EType> ArrayData<EType>
    where
        EType: Clone + std::ops::Add<Output=EType> {
    fn add(&self, other: &ArrayData<EType>) -> Self {
        match (self, other) {
            (ArrayData::CPUArray(ldata), ArrayData::CPUArray(rdata)) => {
                ArrayData::CPUArray(ldata + rdata)
            }
            _ => panic!("cannot add")
        }
    }
}

pub(crate) struct TensorPointer<EType> {
    data: ArrayData<EType>,
    grad: Option<ArrayData<EType>>,
    deps: Vec<Arc<RwLock<TensorPointer<EType>>>>,
}

pub struct Tensor<EType> {
    tp: Arc<RwLock<TensorPointer<EType>>>,
    requires_grad: bool,
}


impl<EType: ArithmeticOps> Tensor<EType> {
    pub fn new<S: AsRef<[usize]>>(shape: S, data: Vec<EType>) -> Result<Tensor<EType>, TensoriaError> {
        let tp = Arc::new(RwLock::new(
            TensorPointer {
                data: ArrayData::new_cpu(shape, data).map_err(|_| TensoriaError::CannotReshapeError {})?,
                grad: None,
                deps: Default::default(),
            }
        ));
        return Ok(Self { tp, requires_grad: false });
    }

    pub fn data(&self) -> ArrayData<EType> {
        let data_ref = &self.tp.read().unwrap().data;
        data_ref.clone()
    }

    fn tensor_add(&self, other: &Tensor<EType>) -> Self {
        let ldata = &self.tp.read().unwrap().data;
        let rdata = &other.tp.read().unwrap().data;
        let res_data = ldata.add(rdata);
        let tp = Arc::new(RwLock::new(TensorPointer {
            data: res_data,
            deps: vec![self.tp.clone(), other.tp.clone()],
        }));
        Self { tp, requires_grad: self.requires_grad || other.requires_grad }
    }
}

impl<EType: ArithmeticOps> Add for &Tensor<EType> {
    type Output = Tensor<EType>;

    fn add(self, rhs: Self) -> Self::Output {
        self.tensor_add(rhs)
    }
}

#[cfg(test)]
mod test {
    use crate::experimental::{ArrayData, Tensor};

    #[test]
    fn simple_add() {
        let x: Tensor<f32> = Tensor::new([1, 2], vec![1., 2.]).unwrap();
        let y: Tensor<f32> = Tensor::new([1, 2], vec![3., 4.]).unwrap();
        let res = &x + &y;
        if let ArrayData::CPUArray(data) = res.data() {
            assert_eq!(data.into_raw_vec(), vec![4., 6.]);
        }
    }
}