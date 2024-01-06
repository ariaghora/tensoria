use std::error::Error;
use std::fmt::Debug;
use std::ops::Add;
use std::sync::{Arc, RwLock};

use bytemuck::Pod;
use ndarray::ArrayD;

use crate::error::TensoriaError;
use crate::gpu::array::{GetType, GPUArray};
use crate::traits::ArithmeticOps;

enum Device { CPU, GPU }


#[derive(Debug)]
enum ArrayData<EType> {
    CPUArray(ArrayD<EType>),
    GPUArray(GPUArray<EType>),
}

impl<EType> ArrayData<EType> {
    pub fn new_cpu<S: AsRef<[usize]>>(shape: S, data: Vec<EType>) -> Result<ArrayData<EType>, Box<dyn Error>> {
        Ok(ArrayData::CPUArray(ArrayD::from_shape_vec(shape.as_ref(), data)?))
    }
}

impl<EType> ArrayData<EType>
    where
        EType: ArithmeticOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    fn arr_add(&self, other: &ArrayData<EType>) -> ArrayData<EType> {
        match (self, other) {
            (ArrayData::CPUArray(ldata), ArrayData::CPUArray(rdata)) => {
                ArrayData::CPUArray(ldata + rdata)
            }
            (ArrayData::GPUArray(ldata_gpu), ArrayData::GPUArray(rdata_gpu)) => {
                ArrayData::GPUArray(ldata_gpu.add(rdata_gpu))
            }
            _ => panic!("cannot add tensors from different device")
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


impl<EType> Tensor<EType>
    where
        EType: ArithmeticOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
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

    pub fn data(&self) -> Vec<EType> {
        let data_ref = &self.tp.read().unwrap().data;
        match data_ref {
            ArrayData::CPUArray(data) => { data.clone().into_raw_vec() }
            ArrayData::GPUArray(data) => { data.data() }
        }
    }

    fn tensor_add(&self, other: &Tensor<EType>) -> Self {
        let ldata = &self.tp.read().unwrap().data;
        let rdata = &other.tp.read().unwrap().data;

        let res_data = ldata.arr_add(rdata);
        
        let tp = Arc::new(RwLock::new(TensorPointer {
            data: res_data,
            deps: vec![self.tp.clone(), other.tp.clone()],
            grad: None,
        }));
        Self { tp, requires_grad: self.requires_grad || other.requires_grad }
    }
}

impl<EType> Add for &Tensor<EType>
    where EType: ArithmeticOps + Clone + Pod + Default + Debug, Vec<EType>: GetType {
    type Output = Tensor<EType>;

    fn add(self, rhs: Self) -> Self::Output {
        self.tensor_add(rhs)
    }
}

#[cfg(test)]
mod test {
    use crate::experimental::Tensor;

    #[test]
    fn simple_add() {
        let x = Tensor::new([1, 2], vec![1., 2.]).unwrap();
        let y = Tensor::new([1, 2], vec![3., 4.]).unwrap();
        let res = &x + &y;
        assert_eq!(res.data(), vec![4., 6.]);
    }
}