use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use bytemuck::Pod;
use ndarray::{ArrayD, Axis, Ix2};

use crate::error::TensoriaError;
use crate::gpu::gpu_array::{GetType, GPUArray};
use crate::traits::TensoriaOps;

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
        EType: TensoriaOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    pub fn new_cpu<S: AsRef<[usize]>>(shape: S, data: Vec<EType>) -> Result<ArrayData<EType>, TensoriaError> {
        Ok(ArrayData::CPUArray(ArrayD::from_shape_vec(shape.as_ref(), data).map_err(|_| TensoriaError::CannotReshapeError {})?))
    }

    pub fn new_gpu<S: AsRef<[usize]>>(shape: S, data: Vec<EType>) -> Result<ArrayData<EType>, TensoriaError> {
        let len = shape.as_ref().iter().fold(1, |x, y| x * y);
        if len != data.len() { return Err(TensoriaError::CannotReshapeError {}); }
        Ok(ArrayData::GPUArray(GPUArray::new(data, shape.as_ref().to_vec())))
    }

    pub fn clone(&self) -> Self {
        match self {
            ArrayData::CPUArray(data) => { Self::CPUArray(data.clone()) }
            ArrayData::GPUArray(_data) => { todo!("GPUArray clone is not implemented yet") }
        }
    }

    pub fn ndim(&self) -> usize {
        match self {
            ArrayData::CPUArray(data) => { data.ndim() }
            ArrayData::GPUArray(data) => { data.shape.len() }
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            ArrayData::CPUArray(data) => { data.shape().to_vec() }
            ArrayData::GPUArray(data) => { data.shape.clone() }
        }
    }
}

/// Following set of implementations are related to public arithmetic functions
impl<EType> ArrayData<EType>
    where
        EType: TensoriaOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    fn arr_add(&self, other: &ArrayData<EType>) -> ArrayData<EType> {
        match (self, other) {
            (ArrayData::CPUArray(ldata), ArrayData::CPUArray(rdata)) => {
                ArrayData::CPUArray(ldata + rdata)
            }
            (ArrayData::GPUArray(ldata), ArrayData::GPUArray(rdata)) => {
                ArrayData::GPUArray(ldata + rdata)
            }
            _ => panic!("cannot add tensors from different device")
        }
    }

    fn arr_div(&self, other: &ArrayData<EType>) -> ArrayData<EType> {
        match (self, other) {
            (ArrayData::CPUArray(ldata), ArrayData::CPUArray(rdata)) => {
                ArrayData::CPUArray(ldata / rdata)
            }
            (ArrayData::GPUArray(_ldata), ArrayData::GPUArray(_rdata)) => {
                todo!();
                // ArrayData::GPUArray(ldata / rdata)
            }
            _ => panic!("cannot add tensors from different device")
        }
    }
    fn arr_mul(&self, other: &ArrayData<EType>) -> ArrayData<EType> {
        match (self, other) {
            (ArrayData::CPUArray(ldata), ArrayData::CPUArray(rdata)) => {
                ArrayData::CPUArray(ldata.mul(rdata))
            }
            (ArrayData::GPUArray(ldata), ArrayData::GPUArray(rdata)) => {
                ArrayData::GPUArray(ldata.mul(rdata))
            }
            _ => panic!("cannot add tensors from different device")
        }
    }
    fn arr_sub(&self, other: &ArrayData<EType>) -> ArrayData<EType> {
        match (self, other) {
            (ArrayData::CPUArray(ldata), ArrayData::CPUArray(rdata)) => {
                ArrayData::CPUArray(ldata.sub(rdata))
            }
            (ArrayData::GPUArray(ldata), ArrayData::GPUArray(rdata)) => {
                ArrayData::GPUArray(ldata.sub(rdata))
            }
            _ => panic!("cannot add tensors from different device")
        }
    }

    pub fn div_scalar_f32(&self, other: f32) -> ArrayData<EType> {
        match self
        {
            ArrayData::CPUArray(data) => {
                let data_scaled = data.map(|v| EType::from(v.to_f32().unwrap() / other).unwrap()).to_owned().into_raw_vec();
                ArrayData::new_cpu(data.shape(), data_scaled).unwrap()
            }
            ArrayData::GPUArray(_) => { todo!() }
        }
    }

    pub fn matmul(&self, other: &ArrayData<EType>) -> ArrayData<EType> {
        let l_ndim = self.ndim();
        let r_ndim = other.ndim();
        if (l_ndim != 2) && (r_ndim != 2) {
            panic!("Both tensors must be of rank-2, but got rank-{} and rank-{} tensors", l_ndim, r_ndim);
        }

        let (l_shape, r_shape) = (self.shape(), other.shape());
        if l_shape[1] != r_shape[0] {
            panic!("Incompatible shape: {:?} and {:?}", l_shape, r_shape);
        }

        match (self, other) {
            (ArrayData::CPUArray(ldata), ArrayData::CPUArray(rdata)) => {
                let ldata_2d = ldata.to_owned().into_dimensionality::<Ix2>().unwrap();
                let rdata_2d = rdata.to_owned().into_dimensionality::<Ix2>().unwrap();

                ArrayData::CPUArray(ldata_2d.dot(&rdata_2d).into_dyn())
            }
            (ArrayData::GPUArray(ldata), ArrayData::GPUArray(rdata)) => {
                ArrayData::GPUArray(ldata.matmul(rdata))
            }
            _ => panic!("cannot add tensors from different device")
        }
    }

    pub fn mean(&self, axis: Option<usize>, keep_dim: bool) -> ArrayData<EType> {
        match self {
            ArrayData::CPUArray(data) => {
                match axis {
                    None => {
                        // TODO: handle scalar rank-0 "array"
                        let mu = data.mean().unwrap();
                        ArrayData::new_cpu([1], vec![mu]).unwrap()
                    }
                    Some(axis) => {
                        let mut mu = data.mean_axis(Axis(axis)).unwrap();
                        if keep_dim { mu = mu.insert_axis(Axis(axis)); }
                        ArrayData::CPUArray(mu)
                    }
                }
            }
            ArrayData::GPUArray(_data) => { todo!("sum is not implemented yet for GPUArray") }
        }
    }

    pub fn sum(&self, axis: Option<usize>, keep_dim: bool) -> ArrayData<EType> {
        match self {
            ArrayData::CPUArray(data) => {
                match axis {
                    None => {
                        // TODO: handle scalar rank-0 "array"
                        let sum = data.sum();
                        ArrayData::new_cpu([1], vec![sum]).unwrap()
                    }
                    Some(axis) => {
                        let mut sum = data.sum_axis(Axis(axis));
                        if keep_dim { sum = sum.insert_axis(Axis(axis)); }
                        ArrayData::CPUArray(sum)
                    }
                }
            }
            ArrayData::GPUArray(_data) => { todo!("sum is not implemented yet for GPUArray") }
        }
    }
    pub fn t(&self) -> ArrayData<EType> {
        if self.ndim() != 2 {
            panic!("Can only transpose a rank-2 tensor, got rank-{}", self.ndim());
        }
        match self {
            ArrayData::CPUArray(data) => {
                ArrayData::CPUArray(data.to_owned().into_dimensionality::<Ix2>().unwrap().t().to_owned().into_dyn())
            }
            ArrayData::GPUArray(_) => { todo!("Transpose GPUArray is not implemented yet") }
        }
    }
}

impl<EType> Add for &ArrayData<EType>
    where
        EType: TensoriaOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    type Output = ArrayData<EType>;

    fn add(self, rhs: Self) -> Self::Output {
        self.arr_add(&rhs)
    }
}

impl<EType> Div for &ArrayData<EType>
    where
        EType: TensoriaOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    type Output = ArrayData<EType>;

    fn div(self, rhs: Self) -> Self::Output {
        self.arr_div(&rhs)
    }
}

impl<EType> Mul for &ArrayData<EType>
    where
        EType: TensoriaOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    type Output = ArrayData<EType>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.arr_mul(&rhs)
    }
}

impl<EType> Sub for &ArrayData<EType>
    where
        EType: TensoriaOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    type Output = ArrayData<EType>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.arr_sub(&rhs)
    }
}
