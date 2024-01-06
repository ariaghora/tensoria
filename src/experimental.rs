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

impl<EType> ArrayData<EType>
    where
        EType: ArithmeticOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    pub fn new_cpu<S: AsRef<[usize]>>(shape: S, data: Vec<EType>) -> Result<ArrayData<EType>, TensoriaError> {
        Ok(ArrayData::CPUArray(ArrayD::from_shape_vec(shape.as_ref(), data).map_err(|_| TensoriaError::CannotReshapeError {})?))
    }

    pub fn new_gpu(shape: Vec<usize>, data: Vec<EType>) -> Result<ArrayData<EType>, TensoriaError> {
        let len = shape.iter().fold(1, |x, y| x * y);
        if len != data.len() { return Err(TensoriaError::CannotReshapeError {}); }
        Ok(ArrayData::GPUArray(GPUArray::new(data, shape)))
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
                data: ArrayData::new_cpu(shape, data)?,
                grad: None,
                deps: Default::default(),
            }
        ));
        return Ok(Self { tp, requires_grad: false });
    }

    pub fn shape(&self) -> Vec<usize> {
        match &self.tp.read().unwrap().data {
            ArrayData::CPUArray(arr) => { arr.shape().to_vec() }
            ArrayData::GPUArray(arr) => { arr.shape.to_vec() }
        }
    }

    pub fn set_requires_grad(&mut self, val: bool) {
        self.requires_grad = val;
        if val {
            // let zero_grad = Self::zeros([2]);
            let shape = self.shape();
            let numel = shape.iter().fold(1, |x, y| x * y);
            let zeros = vec![EType::default(); numel];
            let zero_grad = Some(match &self.tp.read().unwrap().data {
                ArrayData::CPUArray(arr) => {
                    ArrayData::new_cpu(&shape, zeros).unwrap()
                }
                ArrayData::GPUArray(arr) => {
                    ArrayData::new_gpu(shape, zeros).unwrap()
                }
            });

            self.tp.write().unwrap().grad = zero_grad;
        }
    }

    pub fn data(&self) -> Vec<EType> {
        let data_ref = &self.tp.read().unwrap().data;
        match data_ref {
            ArrayData::CPUArray(data) => { data.clone().into_raw_vec() }
            ArrayData::GPUArray(data) => { data.data() }
        }
    }

    /// This will return a Tensor with data located in the GPU.
    /// This operation will detach the tensor from the graph.
    pub fn to_gpu(&self) -> Result<Self, TensoriaError> {
        let data = &self.tp.read().unwrap().data;
        let mut res = match data {
            ArrayData::CPUArray(arr) => {
                let mut new_arr = Self {
                    tp: Arc::new(RwLock::new(TensorPointer {
                        data: ArrayData::new_gpu(arr.shape().to_vec(), arr.as_standard_layout().as_slice().unwrap().to_vec()).unwrap(),
                        grad: None,
                        deps: vec![],
                    })),
                    requires_grad: false,
                };
                new_arr.set_requires_grad(self.requires_grad);
                Ok(new_arr)
            }
            ArrayData::GPUArray(_) => {
                Err(TensoriaError::AlreadyGPUTensor {})
            }
        };
        res
    }
}

impl<EType> Tensor<EType>
    where
        EType: ArithmeticOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    pub fn zeros<Shape: AsRef<[usize]>>(shape: Shape) -> Self {
        let len = shape.as_ref().iter().fold(1, |x, y| x * y);
        Self::new(shape, vec![EType::default(); len]).unwrap()
    }
}

impl<EType> Tensor<EType>
    where
        EType: ArithmeticOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    pub fn tensor_add(&self, other: &Tensor<EType>) -> Self {
        let ldata = &self.tp.read().unwrap().data;
        let rdata = &other.tp.read().unwrap().data;

        let res_data = ldata.arr_add(rdata);

        let tp = Arc::new(RwLock::new(TensorPointer {
            data: res_data,
            deps: vec![self.tp.clone(), other.tp.clone()],
            grad: None,
        }));
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut res = Self { tp, requires_grad: requires_grad };
        res.set_requires_grad(requires_grad);
        res
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
    use crate::error::TensoriaError;
    use crate::experimental::{ArrayData, Tensor};

    #[test]
    fn simple_add() {
        let x = Tensor::new([1, 2], vec![1., 2.]).unwrap();
        let y = Tensor::new([1, 2], vec![3., 4.]).unwrap();
        let res = &x + &y;
        assert_eq!(res.data(), vec![4., 6.]);
    }

    #[test]
    fn simple_add_gpu() -> Result<(), TensoriaError> {
        let x = Tensor::new([1, 2], vec![1., 2.])?.to_gpu()?;
        let y = Tensor::new([1, 2], vec![3., 4.])?.to_gpu()?;
        let res = &x + &y;
        assert_eq!(res.data(), vec![4., 6.]);

        let x = Tensor::new([2], vec![1., 2.])?;
        let y = Tensor::new([2, 1], vec![1., 2.])?;
        let res_cpu = (&x + &y).data();

        let x = Tensor::new([2], vec![1., 2.])?.to_gpu()?;
        let y = Tensor::new([2, 1], vec![1., 2.])?.to_gpu()?;
        let res_gpu = (&x + &y).data();
        assert_eq!(res_cpu, res_gpu);

        Ok(())
    }

    #[test]
    fn requires_grad() {
        let mut x: Tensor<f32> = Tensor::zeros([2, 2]);
        x.set_requires_grad(true);

        assert!(x.tp.read().unwrap().grad.is_some());
        if let ArrayData::CPUArray(arr) = &x.tp.read().unwrap().grad.as_ref().unwrap() {
            assert_eq!(arr.clone().into_raw_vec(), vec![0., 0., 0., 0.])
        } else {
            panic!("This should be CPU array")
        };
    }
}