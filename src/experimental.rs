use std::error::Error;
use std::fmt::Debug;
use std::ops::Add;
use std::sync::{Arc, RwLock};

use bytemuck::Pod;
use ndarray::ArrayD;
use num_traits::NumCast;
use uuid::Uuid;

use crate::error::TensoriaError;
use crate::gpu::array::{GetType, GPUArray};
use crate::traits::ArithmeticOps;

#[derive(PartialEq, Debug)]
enum Device { CPU, GPU }


#[derive(Debug)]
enum ArrayData<EType> {
    CPUArray(ArrayD<EType>),
    GPUArray(GPUArray<EType>),
}

impl<EType> ArrayData<EType> {
    fn device(&self) -> Device {
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
            (ArrayData::GPUArray(ldata_gpu), ArrayData::GPUArray(rdata_gpu)) => {
                // ArrayData::GPUArray(ldata_gpu.mul(rdata_gpu))
                todo!("arr_mul not implemented yet")
            }
            _ => panic!("cannot add tensors from different device")
        }
    }
}

pub(crate) struct TensorPointer<EType> {
    data: ArrayData<EType>,
    grad: Option<ArrayData<EType>>,
    deps: Vec<Arc<RwLock<TensorPointer<EType>>>>,
    backward_fn: Option<BackwardFn<EType>>,
}

pub struct Tensor<EType> {
    // for debugging purpose
    id: Uuid,

    tp: Arc<RwLock<TensorPointer<EType>>>,
    requires_grad: bool,
}

type BackwardFn<EType> = fn(&Vec<Arc<RwLock<TensorPointer<EType>>>>, &Arc<RwLock<TensorPointer<EType>>>);

impl<EType> Tensor<EType>
    where
        EType: ArithmeticOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    pub fn new<S: AsRef<[usize]>>(shape: S, data: Vec<EType>) -> Result<Tensor<EType>, TensoriaError> {
        let tp = Arc::new(RwLock::new(
            TensorPointer {
                data: ArrayData::new_cpu(shape, data)?,
                grad: None,
                backward_fn: None,
                deps: Default::default(),
            }
        ));
        return Ok(Self { id: Uuid::new_v4(), tp, requires_grad: false });
    }

    pub fn backward(&self) -> Result<(), TensoriaError> {
        if !self.requires_grad {
            return Err(TensoriaError::BackwardOnTensorWithNoGrad);
        }

        let shape = self.shape();
        let num_el = shape.iter().fold(1, |x, y| x * y);
        let initial_grad_vec = vec![EType::from(1.).unwrap(); num_el];
        let initial_grad = match self.device() {
            Device::CPU => { ArrayData::new_cpu(shape, initial_grad_vec)? }
            Device::GPU => { ArrayData::new_gpu(shape, initial_grad_vec)? }
        };
        self.tp.write().unwrap().grad = Some(initial_grad);

        self.backward_from_tp(&self.tp);

        Ok(())
    }

    fn backward_from_tp(&self, tp: &Arc<RwLock<TensorPointer<EType>>>) {
        let backward_fn_opt = tp.read().unwrap().backward_fn;
        if let Some(backward_fn) = backward_fn_opt {
            // run backward function for current tensor to obtain gradients of its dependencies
            let deps_arc = &tp
                .read()
                .unwrap()
                .deps;
            let grad_arc = tp;
            backward_fn(deps_arc, grad_arc); // mutating the deps' grad
        }

        // recursively invoke backward on each
        for dep in &tp.read().unwrap().deps {
            self.backward_from_tp(dep)
        }
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
                ArrayData::CPUArray(_) => {
                    ArrayData::new_cpu(&shape, zeros).unwrap()
                }
                ArrayData::GPUArray(_) => {
                    ArrayData::new_gpu(shape, zeros).unwrap()
                }
            });

            self.tp.write().unwrap().grad = zero_grad;
        }
    }

    pub fn data(&self) -> Vec<EType> {
        let data_ref = &self.tp.read().unwrap().data;
        match data_ref {
            ArrayData::CPUArray(data) => { data.to_owned().into_raw_vec() }
            ArrayData::GPUArray(data) => { data.data() }
        }
    }

    pub fn device(&self) -> Device {
        self.tp.read().unwrap().data.device()
    }

    pub fn grad(&self) -> Option<Vec<EType>> {
        let data_ref = &self.tp.read().unwrap().grad;
        match data_ref {
            None => { None }
            Some(arr) => {
                match arr {
                    ArrayData::CPUArray(val) => { Some(val.to_owned().into_raw_vec()) }
                    ArrayData::GPUArray(val) => { Some(val.data()) }
                }
            }
        }
    }

    /// This will return a Tensor with data located in the GPU.
    /// This operation will detach the tensor from the graph.
    pub fn to_gpu(&self) -> Result<Self, TensoriaError> {
        let data = &self.tp.read().unwrap().data;
        let mut res = match data {
            ArrayData::CPUArray(arr) => {
                let mut new_arr = Self {
                    id: self.id,
                    tp: Arc::new(RwLock::new(TensorPointer {
                        data: ArrayData::new_gpu(arr.shape().to_vec(), arr.as_standard_layout().as_slice().unwrap().to_vec()).unwrap(),
                        grad: None,
                        backward_fn: self.tp.read().unwrap().backward_fn,
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

type BinopGradUpdateFn<T> = fn(l_arr: &ArrayData<T>, r_arr: &ArrayData<T>, out_grad: &ArrayData<T>) -> ArrayData<T>;

impl<EType> Tensor<EType>
    where
        EType: ArithmeticOps + Clone + Pod + Default + Debug,
        Vec<EType>: GetType {
    pub fn tensor_add(&self, other: &Tensor<EType>) -> Self {
        let l_grad_update_fn: BinopGradUpdateFn<EType> = |l, r, g| {
            l.arr_add(g)
        };
        let r_grad_update_fn: BinopGradUpdateFn<EType> = |l, r, g| {
            r.arr_add(g)
        };
        self.tensor_binop(other, l_grad_update_fn, r_grad_update_fn)
    }

    pub fn tensor_binop(
        &self, other: &Tensor<EType>, l_grad_update_fn: BinopGradUpdateFn<EType>, r_grad_update_fn: BinopGradUpdateFn<EType>,
    ) -> Self {
        let ldata = &self.tp.read().unwrap().data;
        let rdata = &other.tp.read().unwrap().data;
        let res_data = ldata.arr_add(rdata);

        let requires_grad = self.requires_grad || other.requires_grad;

        let bacward_fn: Option<BackwardFn<EType>> = if requires_grad {
            let func = |deps: &Vec<Arc<RwLock<TensorPointer<EType>>>>, tensor: &Arc<RwLock<TensorPointer<EType>>>| {
                let new_l_grad = {
                    let grad_arr_opt = &tensor.read().unwrap().grad;
                    let l_grad_opt = &deps[0].read().unwrap().grad;
                    if let (Some(l_arr), Some(grad_arr)) = (l_grad_opt, grad_arr_opt) {
                        // TODO: this should generalize to other binary op
                        Some(l_arr.arr_add(grad_arr))
                    } else {
                        None
                    }
                };
                deps[0].write().unwrap().grad = new_l_grad;

                let new_r_grad = {
                    let grad_arr_opt = &tensor.read().unwrap().grad;
                    let r_grad_opt = &deps[1].read().unwrap().grad;
                    if let (Some(r_arr), Some(grad_arr)) = (r_grad_opt, grad_arr_opt) {
                        // TODO: this should generalize to other binary op
                        Some(r_arr.arr_add(grad_arr))
                    } else {
                        None
                    }
                };
                deps[1].write().unwrap().grad = new_r_grad;
            };

            Some(func)
        } else {
            None
        };

        let tp = Arc::new(RwLock::new(TensorPointer {
            data: res_data,
            deps: vec![self.tp.clone(), other.tp.clone()],
            backward_fn: bacward_fn,
            grad: None,
        }));


        let mut res = Self { id: Uuid::new_v4(), tp, requires_grad };
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
    use std::ops::Add;

    use crate::error::TensoriaError;
    use crate::experimental::{ArrayData, Device, Tensor};

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
        assert_eq!(x.device(), Device::CPU);

        let x = Tensor::new([2], vec![1., 2.])?.to_gpu()?;
        let y = Tensor::new([2, 1], vec![1., 2.])?.to_gpu()?;
        let res_gpu = (&x + &y).data();
        assert_eq!(x.device(), Device::GPU);
        assert_eq!(res_cpu, res_gpu);

        Ok(())
    }

    #[test]
    fn requires_grad() {
        let mut x: Tensor<f32> = Tensor::zeros([2, 2]);
        x.set_requires_grad(true);

        assert!(x.tp.read().unwrap().grad.is_some());
        if let ArrayData::CPUArray(arr) = &x.tp.read().unwrap().grad.as_ref().unwrap() {
            assert_eq!(arr.to_owned().into_raw_vec(), vec![0., 0., 0., 0.])
        } else {
            panic!("This should be CPU array")
        };
    }

    #[test]
    fn backward() {
        let mut x = Tensor::new([2], vec![1., 2.]).unwrap();
        x.set_requires_grad(true);
        let y = Tensor::new([2], vec![3., 4.]).unwrap();
        let res = x.add(&y);

        assert_eq!(res.tp.read().unwrap().deps.len(), 2);

        res.backward().unwrap();

        assert_eq!(res.grad(), Some(vec![1., 1.]));
        assert_eq!(y.grad(), None);
        assert_eq!(x.grad(), Some(vec![1., 1.]));

        let mut x = Tensor::new([2], vec![1., 2.]).unwrap();
        x.set_requires_grad(true);
        let res = &x + &x;
        res.backward().unwrap();
        assert_eq!(x.grad(), Some(vec![2., 2.]));
    }
}