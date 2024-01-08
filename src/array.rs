use std::fmt::Debug;

use bytemuck::Pod;
use ndarray::ArrayD;

use crate::error::TensoriaError;
use crate::gpu::gpu_array::{GetType, GPUArray};
use crate::traits::ArithmeticOps;

#[derive(PartialEq, Debug)]
pub enum Device { CPU, GPU }


#[derive(Debug)]
pub enum ArrayData<EType> {
    CPUArray(ArrayD<EType>),
    GPUArray(GPUArray<EType>),
}

impl<EType> ArrayData<EType> {
    pub(crate) fn device(&self) -> Device {
        match self {
            ArrayData::CPUArray(_) => Device::CPU,
            ArrayData::GPUArray(_) => Device::GPU
        }
    }
}

impl<EType> ArrayData<EType>
    where
        EType: ArithmeticOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    pub fn new_cpu<S: AsRef<[usize]>>(shape: S, data: Vec<EType>) -> Result<ArrayData<EType>, TensoriaError> {
        Ok(ArrayData::CPUArray(ArrayD::from_shape_vec(shape.as_ref(), data).map_err(|_| TensoriaError::CannotReshapeError {})?))
    }

    pub fn new_gpu<S: AsRef<[usize]>>(shape: S, data: Vec<EType>) -> Result<ArrayData<EType>, TensoriaError> {
        let len = shape.as_ref().iter().fold(1, |x, y| x * y);
        if len != data.len() { return Err(TensoriaError::CannotReshapeError {}); }
        Ok(ArrayData::GPUArray(GPUArray::new(data, shape.as_ref().to_vec())))
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
    fn arr_mul(&self, other: &ArrayData<EType>) -> ArrayData<EType> {
        match (self, other) {
            (ArrayData::CPUArray(ldata), ArrayData::CPUArray(rdata)) => {
                ArrayData::CPUArray(ldata * rdata)
            }
            (ArrayData::GPUArray(ldata), ArrayData::GPUArray(rdata)) => {
                ArrayData::GPUArray(ldata.mul(rdata))
            }
            _ => panic!("cannot add tensors from different device")
        }
    }
}

impl<EType> std::ops::Add for &ArrayData<EType>
    where
        EType: ArithmeticOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    type Output = ArrayData<EType>;

    fn add(self, rhs: Self) -> Self::Output {
        self.arr_add(&rhs)
    }
}

impl<EType> std::ops::Mul for &ArrayData<EType>
    where
        EType: ArithmeticOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    type Output = ArrayData<EType>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.arr_mul(&rhs)
    }
}
