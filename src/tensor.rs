use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::RwLockReadGuard;
use std::sync::{Arc, RwLock};

use bytemuck::Pod;
use uuid::Uuid;

use crate::array::{ArrayData, Device};
use crate::error::TensoriaError;
use crate::gpu::gpu_array::GetType;
use crate::traits::TensoriaOps;

pub struct TensorPointer<EType> {
    pub(crate) data: ArrayData<EType>,
    grad: Option<ArrayData<EType>>,
    deps: Vec<Arc<RwLock<TensorPointer<EType>>>>,
    grad_fn: Option<GradFn<EType>>,
}

pub struct Tensor<EType> {
    // for debugging purpose
    id: Uuid,

    tp: Arc<RwLock<TensorPointer<EType>>>,
    requires_grad: bool,
}

pub(crate) type UnOpFn<EType> = Box<dyn FnOnce(&ArrayData<EType>) -> ArrayData<EType>>;
pub(crate) type BinOpFn<EType> =
    Box<dyn FnOnce(&ArrayData<EType>, &ArrayData<EType>) -> ArrayData<EType>>;
pub(crate) type GradFn<EType> = fn(
    old_grad: &ArrayData<EType>,
    parent_grad: &ArrayData<EType>,
    parent: &Arc<RwLock<TensorPointer<EType>>>,
) -> ArrayData<EType>;

impl<EType> Tensor<EType>
where
    EType: TensoriaOps + Clone + Pod + Default + Debug,
    Vec<EType>: GetType,
{
    pub fn new<S: AsRef<[usize]>>(
        shape: S,
        data: Vec<EType>,
    ) -> Result<Tensor<EType>, TensoriaError> {
        let tp = Arc::new(RwLock::new(TensorPointer {
            data: ArrayData::new_cpu(shape, data)?,
            grad: None,
            grad_fn: None,
            deps: Default::default(),
        }));
        return Ok(Self {
            id: Uuid::new_v4(),
            tp,
            requires_grad: false,
        });
    }

    pub fn new_gpu<S: AsRef<[usize]>>(
        shape: S,
        data: Vec<EType>,
    ) -> Result<Tensor<EType>, TensoriaError> {
        let tp = Arc::new(RwLock::new(TensorPointer {
            data: ArrayData::new_gpu(shape, data)?,
            grad: None,
            grad_fn: None,
            deps: Default::default(),
        }));
        return Ok(Self {
            id: Uuid::new_v4(),
            tp,
            requires_grad: false,
        });
    }

    pub fn backward(&self) -> Result<(), TensoriaError> {
        if !self.requires_grad {
            return Err(TensoriaError::BackwardOnTensorWithNoGrad);
        }

        let shape = self.shape();
        let num_el = shape.iter().fold(1, |x, y| x * y);
        let initial_grad_vec = vec![EType::from(1.).unwrap(); num_el];
        let initial_grad = match self.device() {
            Device::CPU => ArrayData::new_cpu(shape, initial_grad_vec)?,
            Device::GPU => ArrayData::new_gpu(shape, initial_grad_vec)?,
        };
        self.tp.write().unwrap().grad = Some(initial_grad);

        self.backward_from_tp(&self.tp);

        Ok(())
    }

    fn backward_from_tp(&self, tp: &Arc<RwLock<TensorPointer<EType>>>) {
        // recursively invoke backward on each
        let parent_grad_opt = &tp.read().unwrap().grad;

        for dep in &tp.read().unwrap().deps {
            let new_grad = {
                let grad_fn_opt = &dep.read().unwrap().grad_fn;
                if let (Some(old_grad), Some(parent_grad), Some(grad_fn)) =
                    (&dep.read().unwrap().grad, parent_grad_opt, grad_fn_opt)
                {
                    // calculate new grad for dep
                    let new_grad = grad_fn(old_grad, parent_grad, tp);
                    Some(new_grad)
                } else {
                    None
                }
            };
            dep.write().unwrap().grad = new_grad;
            self.backward_from_tp(dep);
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match &self.tp.read().unwrap().data {
            ArrayData::CPUArray(arr) => arr.shape().to_vec(),
            ArrayData::GPUArray(arr) => arr.shape.to_vec(),
        }
    }

    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.set_requires_grad(true)
        }
    }

    pub fn set_requires_grad(&mut self, val: bool) {
        self.requires_grad = val;
        if val {
            let shape = self.shape();
            let numel = shape.iter().fold(1, |x, y| x * y);
            let zeros = vec![EType::default(); numel];
            let zero_grad = Some(match &self.tp.read().unwrap().data {
                ArrayData::CPUArray(_) => ArrayData::new_cpu(&shape, zeros).unwrap(),
                ArrayData::GPUArray(_) => ArrayData::new_gpu(shape, zeros).unwrap(),
            });

            self.tp.write().unwrap().grad = zero_grad;
        } else {
            self.tp.write().unwrap().grad = None;
        }
    }

    /// Fetch ArrayData<EType> with device type corresponding to Self. This method
    /// performs data copy.
    pub fn data(&self) -> ArrayData<EType> {
        self.tp.read().unwrap().data.clone()
    }

    pub fn to_vec(&self) -> Vec<EType> {
        let data_ref = &self.tp.read().unwrap().data;
        match data_ref {
            ArrayData::CPUArray(data) => data.to_owned().into_raw_vec(),
            ArrayData::GPUArray(data) => data.to_vec(),
        }
    }

    pub fn device(&self) -> Device {
        self.tp.read().unwrap().data.device()
    }

    pub fn grad(&self) -> Option<ArrayData<EType>> {
        match &self.tp.read().unwrap().grad {
            None => None,
            Some(data) => Some(data.clone()),
        }
    }

    pub fn grad_vec(&self) -> Option<Vec<EType>> {
        let data_ref = &self.tp.read().unwrap().grad;
        match data_ref {
            None => None,
            Some(arr) => match arr {
                ArrayData::CPUArray(val) => Some(val.to_owned().into_raw_vec()),
                ArrayData::GPUArray(val) => Some(val.to_vec()),
            },
        }
    }

    /// This will return a Tensor with data located in the GPU.
    /// This operation will detach the tensor from the graph.
    pub fn to_gpu(&self) -> Result<Self, TensoriaError> {
        let data = &self.tp.read().unwrap().data;
        let res = match data {
            ArrayData::CPUArray(arr) => {
                let mut new_arr = Self {
                    id: self.id,
                    tp: Arc::new(RwLock::new(TensorPointer {
                        data: ArrayData::new_gpu(
                            arr.shape().to_vec(),
                            arr.as_standard_layout().as_slice().unwrap().to_vec(),
                        )
                        .unwrap(),
                        grad: None,
                        grad_fn: self.tp.read().unwrap().grad_fn,
                        deps: vec![],
                    })),
                    requires_grad: false,
                };

                new_arr.set_requires_grad(self.requires_grad);
                Ok(new_arr)
            }
            ArrayData::GPUArray(_) => Err(TensoriaError::AlreadyGPUTensor {}),
        };
        res
    }

    pub fn update_data<F: Fn(&ArrayData<EType>) -> ArrayData<EType>>(&self, update_fn: F) {
        let new_data = update_fn(&self.tp.read().unwrap().data);
        self.tp.write().unwrap().data = new_data;
    }
}

impl<EType> Tensor<EType>
where
    EType: TensoriaOps + Clone + Pod + Default + Debug,
    Vec<EType>: GetType,
{
    pub fn zeros<Shape: AsRef<[usize]>>(_shape: Shape) -> Self {
        todo!()
    }
}

impl<EType> Tensor<EType>
where
    EType: TensoriaOps + Clone + Pod + Default + Debug,
    Vec<EType>: GetType,
{
    pub fn matmul(&self, other: &Tensor<EType>) -> Self {
        let lgf: Option<GradFn<EType>> = Some(|lg, og, parent| {
            let parent = &parent.read().unwrap();
            let rhs = &parent.deps[1].read().unwrap().data;
            lg.add(&og.matmul(&rhs.t()))
        });
        let rgf: Option<GradFn<EType>> = Some(|rg, og, parent| {
            let parent = &parent.read().unwrap();
            let lhs = &parent.deps[0].read().unwrap().data;
            rg.add(&lhs.t().matmul(&og))
        });
        let add_fn: BinOpFn<EType> = Box::new(|a, b| a.matmul(&b));
        self.tensor_binop(other, add_fn, lgf, rgf)
    }
    pub fn tensor_add(&self, other: &Tensor<EType>) -> Self {
        let lgf: Option<GradFn<EType>> = Some(|lg, og, _| {
            let og = gradient_broadcasting(lg, og);
            lg.add(og)
        });
        let rgf: Option<GradFn<EType>> = Some(|rg, og, _| {
            let og = gradient_broadcasting(rg, og);
            rg.add(&og)
        });
        let add_fn: BinOpFn<EType> = Box::new(|a, b| a.add(b));
        self.tensor_binop(other, add_fn, lgf, rgf)
    }

    pub fn tensor_mul(&self, other: &Tensor<EType>) -> Self {
        let lgf: Option<GradFn<EType>> = Some(|lg, og, parent| {
            let og = gradient_broadcasting(lg, og);
            let parent = &parent.read().unwrap();
            let rhs = &parent.deps[1].read().unwrap().data;
            lg.add(&rhs.mul(&og))
        });
        let rgf: Option<GradFn<EType>> = Some(|rg, og, parent| {
            let og = gradient_broadcasting(rg, og);
            let parent = &parent.read().unwrap();
            let lhs = &parent.deps[0].read().unwrap().data;
            rg.add(&lhs.mul(&og))
        });
        let mul_fn: BinOpFn<EType> = Box::new(|a, b| a.mul(b));
        self.tensor_binop(other, mul_fn, lgf, rgf)
    }

    pub fn tensor_div(&self, other: &Tensor<EType>) -> Self {
        let lgf: Option<GradFn<EType>> = Some(|lg, og, parent| {
            let parent = &parent.read().unwrap();
            let rhs = &parent.deps[1].read().unwrap().data;
            let local_grad = &og.div(rhs);
            lg.add(gradient_broadcasting(&lg, local_grad))
        });
        let rgf: Option<GradFn<EType>> = Some(|rg, og, parent| {
            let parent = &parent.read().unwrap();
            let lhs = &parent.deps[0].read().unwrap().data;
            let rhs = &parent.deps[1].read().unwrap().data;
            let local_grad = &lhs
                .mul(og.scale(EType::from(-1).unwrap()))
                .div(&rhs.powi(2));
            rg.add(gradient_broadcasting(&rg, local_grad))
        });
        let div_fn: BinOpFn<EType> = Box::new(|a, b| a.div(b));
        self.tensor_binop(other, div_fn, lgf, rgf)
    }

    pub fn tensor_sub(&self, other: &Tensor<EType>) -> Self {
        let lgf: Option<GradFn<EType>> = Some(|lg, og, _| {
            let og = gradient_broadcasting(lg, og);
            lg.add(&og)
        });
        let rgf: Option<GradFn<EType>> = Some(|rg, og, _| {
            let og = gradient_broadcasting(rg, og);
            rg.sub(&og)
        });
        let sub_fn: BinOpFn<EType> = Box::new(|a, b| a.sub(b));
        self.tensor_binop(other, sub_fn, lgf, rgf)
    }

    pub fn exp(&self) -> Self {
        let exp_fn: UnOpFn<EType> = Box::new(move |data| data.exp());
        let gf: Option<GradFn<EType>> = Some(|g, og, parent| {
            let parent = &parent.read().unwrap();
            let x = &parent.deps[0].read().unwrap().data;
            g.add(&og.mul(x.exp()))
        });
        self.tensor_unop(exp_fn, gf)
    }

    pub fn mean(&self, axis: Option<usize>, keep_dim: bool) -> Self {
        let mean_fn: UnOpFn<EType> = Box::new(move |data| data.mean(axis, keep_dim));

        let gf: Option<GradFn<EType>> = Some(move |g, og, parent| {
            let shape = g.shape();

            // Get axis data, that is the second dependency of the parent, i.e., deps[1].
            // it will be -1 if no axis is specified and greater than or equals to zero otherwise.
            let axis = match &parent.read().unwrap().deps[1].read().unwrap().data {
                ArrayData::CPUArray(ax) => ax.first().unwrap().to_owned(),
                ArrayData::GPUArray(ax) => ax.to_vec()[0],
            };

            let axis_is_specified = axis > EType::from(-1).unwrap();
            let numel = if axis_is_specified {
                shape[axis.to_usize().unwrap()]
            } else {
                g.shape().iter().fold(1, |x, y| x * y)
            };

            let og_mean = og.div_scalar_f32(numel as f32);
            if axis_is_specified {
                let og_mean_broadcast = gradient_broadcasting(g, &og_mean);
                return g.add(&og_mean_broadcast);
            } else {
                return g.add(&og_mean);
            }
        });
        let res = self.tensor_unop(mean_fn, gf);

        // Hacky way to pass axis info to tensor's dependency, since GradFn is a function (not a closure)
        // and we want to avoid capturing outer scope of `gf`.
        let t_axis = match axis {
            None => Self::new([1], vec![EType::from(-1).unwrap()]),
            Some(axis) => Self::new([1], vec![EType::from(axis).unwrap()]),
        }
        .unwrap();
        res.tp.write().unwrap().deps.push(t_axis.tp);
        res
    }

    pub fn sum(&self, axis: Option<usize>, keep_dim: bool) -> Self {
        let sum_fn: UnOpFn<EType> = Box::new(move |data| data.sum(axis, keep_dim));

        let gf: Option<GradFn<EType>> = Some(|g, og, parent| {
            let axis = match &parent.read().unwrap().deps[1].read().unwrap().data {
                ArrayData::CPUArray(ax) => ax.first().unwrap().to_owned(),
                ArrayData::GPUArray(ax) => ax.to_vec()[0],
            };
            let axis_is_specified = axis > EType::from(-1).unwrap();
            if axis_is_specified {
                // The gradient of sum is broadcasting the gradient back to the shape of 'g'.
                let og_broadcast = gradient_broadcasting(g, &og);
                return g.add(&og_broadcast);
            } else {
                return g.add(og);
            }
        });
        let res = self.tensor_unop(sum_fn, gf);

        // Hacky way to pass axis info to tensor's dependency, since GradFn is a function (not a closure)
        // and we want to avoid capturing outer scope of `gf`.
        let t_axis = match axis {
            None => Self::new([1], vec![EType::from(-1).unwrap()]),
            Some(axis) => Self::new([1], vec![EType::from(axis).unwrap()]),
        }
        .unwrap();
        res.tp.write().unwrap().deps.push(t_axis.tp);
        res
    }

    pub(crate) fn tensor_binop(
        &self,
        other: &Tensor<EType>,
        binop_fn: BinOpFn<EType>,
        l_grad_fn: Option<GradFn<EType>>,
        r_grad_fn: Option<GradFn<EType>>,
    ) -> Self {
        self.tp.write().unwrap().grad_fn = l_grad_fn;
        other.tp.write().unwrap().grad_fn = r_grad_fn;

        let ldata = &self.tp.read().unwrap().data;
        let rdata = &other.tp.read().unwrap().data;
        let res_data = binop_fn(ldata, rdata);

        let requires_grad = self.requires_grad || other.requires_grad;

        let tp = Arc::new(RwLock::new(TensorPointer {
            data: res_data,
            deps: vec![self.tp.clone(), other.tp.clone()],
            grad: None,
            grad_fn: None,
        }));

        let mut res = Self {
            id: Uuid::new_v4(),
            tp,
            requires_grad,
        };
        res.set_requires_grad(requires_grad);
        res
    }

    pub(crate) fn tensor_unop(
        &self,
        unop_fn: UnOpFn<EType>,
        grad_fn: Option<GradFn<EType>>,
    ) -> Self {
        self.tp.write().unwrap().grad_fn = grad_fn;

        let data = &self.tp.read().unwrap().data;
        let res_data = unop_fn(data);

        let requires_grad = self.requires_grad;

        let tp = Arc::new(RwLock::new(TensorPointer {
            data: res_data,
            deps: vec![self.tp.clone()],
            grad: None,
            grad_fn: None,
        }));

        let mut res = Self {
            id: Uuid::new_v4(),
            tp,
            requires_grad,
        };
        res.set_requires_grad(requires_grad);
        res
    }
}

fn gradient_broadcasting<EType>(
    self_grad: &ArrayData<EType>,
    out_grad: &ArrayData<EType>,
) -> ArrayData<EType>
where
    EType: TensoriaOps + Clone + Pod + Default + Debug,
    Vec<EType>: GetType,
{
    // Sum out added dims
    let mut out_grad = out_grad.clone();
    let ndims_added = out_grad.ndim() as i32 - self_grad.ndim() as i32;
    for _ in 0..ndims_added {
        out_grad = out_grad.sum(Some(0), false);
    }

    // Sum across broadcasted but non-added dims
    for (i, dim) in self_grad.shape().iter().enumerate() {
        if dim == &1 {
            out_grad = out_grad.sum(Some(i), true);
        }
    }
    out_grad
}

/// Macro for several binary operators, so it is easier to implement it for both
/// op(Self, &Self) and op(&Self, &Self)
macro_rules! impl_bin_op {
    ($trait:ident, $trait_method:ident, $tensor_method:ident) => {
        impl<EType> std::ops::$trait<&Self> for Tensor<EType>
        where
            EType: TensoriaOps + Clone + Pod + Default + Debug,
            Vec<EType>: GetType,
        {
            type Output = Tensor<EType>;

            fn $trait_method(self, rhs: &Self) -> Self::Output {
                self.$tensor_method(rhs)
            }
        }

        impl<EType> std::ops::$trait for &Tensor<EType>
        where
            EType: TensoriaOps + Clone + Pod + Default + Debug,
            Vec<EType>: GetType,
        {
            type Output = Tensor<EType>;

            fn $trait_method(self, rhs: Self) -> Self::Output {
                self.$tensor_method(rhs)
            }
        }

        impl<EType> std::ops::$trait<Tensor<EType>> for &Tensor<EType>
        where
            EType: TensoriaOps + Clone + Pod + Default + Debug,
            Vec<EType>: GetType,
        {
            type Output = Tensor<EType>;

            fn $trait_method(self, rhs: Tensor<EType>) -> Self::Output {
                self.$tensor_method(&rhs)
            }
        }

        impl<EType> std::ops::$trait for Tensor<EType>
        where
            EType: TensoriaOps + Clone + Pod + Default + Debug,
            Vec<EType>: GetType,
        {
            type Output = Tensor<EType>;

            fn $trait_method(self, rhs: Self) -> Self::Output {
                self.$tensor_method(&rhs)
            }
        }

        impl<'a, EType> std::ops::$trait<Tensor<EType>> for RwLockReadGuard<'a, Tensor<EType>>
        where
            EType: TensoriaOps + Clone + Pod + Default + Debug,
            Vec<EType>: GetType,
        {
            type Output = Tensor<EType>;

            fn $trait_method(self, rhs: Tensor<EType>) -> Self::Output {
                self.$tensor_method(&rhs)
            }
        }

        impl<'a, EType> std::ops::$trait<&RwLockReadGuard<'a, Tensor<EType>>> for Tensor<EType>
        where
            EType: TensoriaOps + Clone + Pod + Default + Debug,
            Vec<EType>: GetType,
        {
            type Output = Tensor<EType>;

            fn $trait_method(self, rhs: &RwLockReadGuard<'a, Tensor<EType>>) -> Self::Output {
                self.$tensor_method(&rhs)
            }
        }
    };
}

impl_bin_op!(Add, add, tensor_add);
impl_bin_op!(Div, div, tensor_div);
impl_bin_op!(Mul, mul, tensor_mul);
impl_bin_op!(Sub, sub, tensor_sub);

/// The suites are solely to test tensor's autograd mechanism
#[cfg(test)]
mod test {
    use std::ops::Sub;

    use crate::error::TensoriaError;
    use crate::tensor::{Device, Tensor};

    #[test]
    fn add() -> Result<(), TensoriaError> {
        let x = Tensor::new([2, 2], vec![1., 2., 3., 4.])?;
        let mut y = Tensor::new([2], vec![1., 1.])?;
        y.set_requires_grad(true);
        (&x + &y).backward()?;
        assert_eq!(y.grad_vec(), Some(vec![2., 2.]));
        Ok(())
    }

    #[test]
    fn add_gpu() -> Result<(), TensoriaError> {
        let x = Tensor::new([1, 2], vec![1., 2.])?.to_gpu()?;
        let mut y = Tensor::new([1, 2], vec![3., 4.])?.to_gpu()?;
        y.set_requires_grad(true);

        let res = &x + &y;
        assert_eq!(res.to_vec(), vec![4., 6.]);
        res.backward()?;
        assert_eq!(y.grad_vec(), Some(vec![1., 1.]));
        Ok(())
    }

    #[test]
    fn div() -> Result<(), TensoriaError> {
        let mut x = Tensor::new([2, 2], vec![1., 2., 3., 4.])?;
        let mut y = Tensor::new([2], vec![1., 2.])?;
        x.set_requires_grad(true);
        y.set_requires_grad(true);
        let res = &x / &y;
        assert_eq!(res.to_vec(), vec![1., 1., 3., 2.]);
        res.backward()?;
        assert_eq!(x.grad_vec(), Some(vec![1., 0.5, 1., 0.5]));
        assert_eq!(y.grad_vec(), Some(vec![-4., -1.5]));
        Ok(())
    }

    #[test]
    fn sub() -> Result<(), TensoriaError> {
        let x = Tensor::new([1, 2], vec![1., 2.])?.to_gpu()?;
        let y = Tensor::new([1, 2], vec![3., 4.])?.to_gpu()?;
        let res = &x - &y;
        assert_eq!(res.to_vec(), vec![-2., -2.]);

        let x = Tensor::new([2], vec![1., 2.])?;
        let y = Tensor::new([2, 1], vec![1., 2.])?;
        let res_cpu = (&x - &y).to_vec();
        assert_eq!(x.device(), Device::CPU);

        let x = Tensor::new([2], vec![1., 2.])?.to_gpu()?;
        let y = Tensor::new([2, 1], vec![1., 2.])?.to_gpu()?;
        let res_gpu = (&x - &y).to_vec();
        assert_eq!(x.device(), Device::GPU);
        assert_eq!(res_cpu, res_gpu);

        Ok(())
    }

    #[test]
    fn sub_gpu() -> Result<(), TensoriaError> {
        let x = Tensor::new([2], vec![1., 2.])?.to_gpu()?;
        let mut y = Tensor::new([2, 1], vec![1., 2.])?.to_gpu()?;
        y.set_requires_grad(true);
        let res_gpu = &x - &y;
        res_gpu.backward()?;
        assert_eq!(x.device(), Device::GPU);
        assert_eq!(y.grad_vec(), Some(vec![-2., -2.]));

        let x = Tensor::new([2, 2], vec![1., 2., 3., 4.])?.to_gpu()?;
        let mut y = Tensor::new([2, 2], vec![1., 2., 3., 4.])?.to_gpu()?;
        y.set_requires_grad(true);
        (x.sub(&y)).backward()?;
        assert_eq!(y.grad_vec(), Some(vec![-1.; 4]));

        Ok(())
    }

    #[test]
    fn mul() -> Result<(), TensoriaError> {
        let mut x = Tensor::new([2, 2], vec![1, 2, 3, 4])?;
        x.set_requires_grad(true);

        let y = Tensor::new([2, 2], vec![2, 3, 4, 5])?;
        let res = &x * y;

        res.backward()?;
        assert_eq!(x.grad_vec().unwrap(), vec![2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn mul_gpu() -> Result<(), TensoriaError> {
        let mut x = Tensor::new([2, 2], vec![1, 2, 3, 4])?;
        x.set_requires_grad(true);
        (&(&x * &x) * &x).backward()?;
        assert_eq!(x.grad_vec().unwrap(), vec![3, 12, 27, 48]);

        let mut x = Tensor::new([2, 2], vec![1, 2, 3, 4])?.to_gpu()?;
        x.set_requires_grad(true);
        (&(&x * &x) * &x).backward()?;
        assert_eq!(x.grad_vec().unwrap(), vec![3, 12, 27, 48]);
        Ok(())
    }

    #[test]
    fn matmul() -> Result<(), TensoriaError> {
        let mut x = Tensor::new([2, 2], vec![1, 2, 3, 4])?;
        x.set_requires_grad(true);

        let mut y = Tensor::new([2, 1], vec![1, 2])?;
        y.set_requires_grad(true);

        let res = x.matmul(&y);
        assert_eq!(res.to_vec(), vec![5, 11]);

        res.backward()?;
        assert_eq!(x.grad_vec(), Some(vec![1, 2, 1, 2]));
        assert_eq!(y.grad_vec(), Some(vec![4, 6]));
        Ok(())
    }

    #[test]
    fn mean() -> Result<(), TensoriaError> {
        let mut x = Tensor::new([3], vec![4., 2., 3.])?;
        x.set_requires_grad(true);
        let res = x.mean(None, false);
        res.backward()?;
        assert_eq!(res.to_vec(), vec![3.]);
        assert_eq!(x.grad_vec(), Some(vec![1. / 3., 1. / 3., 1. / 3.]));

        let mut x = Tensor::new([2, 3], vec![1.; 6])?;
        x.set_requires_grad(true);
        let res = x.mean(None, false);
        res.backward()?;
        assert_eq!(res.to_vec(), vec![1.]);
        assert_eq!(x.grad_vec(), Some(vec![1. / 6.; 6]));

        let mut x = Tensor::new([2, 3], vec![2., 2., 2., 2., 2., 2.])?;
        x.set_requires_grad(true);
        let res = x.mean(Some(0), true);
        res.backward()?;
        assert_eq!(res.shape(), vec![1, 3]);
        assert_eq!(x.grad_vec(), Some(vec![0.5; 6]));

        x.zero_grad();
        let res = x.mean(Some(1), true);
        res.backward()?;
        assert_eq!(x.grad_vec(), Some(vec![1. / 3.; 6]));

        Ok(())
    }

    #[test]
    fn mean_gpu() -> Result<(), TensoriaError> {
        let mut x = Tensor::new([3], vec![4., 2., 3.])?.to_gpu()?;
        x.set_requires_grad(true);
        let res = x.mean(None, false);
        res.backward()?;
        assert_eq!(res.to_vec(), vec![3.]);
        assert_eq!(x.grad_vec(), Some(vec![1. / 3., 1. / 3., 1. / 3.]));

        let mut x = Tensor::new([2, 3], vec![2., 2., 2., 2., 2., 2.])?.to_gpu()?;
        x.set_requires_grad(true);
        let res = x.mean(Some(0), true);
        res.backward()?;
        assert_eq!(res.shape(), vec![1, 3]);
        assert_eq!(x.grad_vec(), Some(vec![0.5; 6]));

        Ok(())
    }

    #[test]
    fn sum() -> Result<(), TensoriaError> {
        let mut x = Tensor::new([1, 3], vec![1., 2., 3.])?;
        x.set_requires_grad(true);
        let res = x.sum(None, false);
        res.backward()?;
        assert_eq!(res.shape(), vec![1]);
        assert_eq!(x.grad_vec(), Some(vec![1.; 3]));

        let mut x = Tensor::new([1, 3], vec![1., 2., 3.])?;
        x.set_requires_grad(true);
        let res = x.sum(Some(1), false);
        res.backward()?;
        assert_eq!(res.shape(), vec![1]);
        assert_eq!(x.grad_vec(), Some(vec![1.; 3]));
        Ok(())
    }

    #[test]
    fn sum_gpu() -> Result<(), TensoriaError> {
        let mut x = Tensor::new([1, 3], vec![1., 2., 3.])?.to_gpu()?;
        x.set_requires_grad(true);
        let res = x.sum(None, false);
        res.backward()?;
        assert_eq!(res.shape(), vec![1]);
        assert_eq!(x.grad_vec(), Some(vec![1.; 3]));

        let mut x = Tensor::new([1, 3], vec![1., 2., 3.])?.to_gpu()?;
        x.set_requires_grad(true);
        let res = x.sum(Some(1), false);
        res.backward()?;
        assert_eq!(res.shape(), vec![1]);
        assert_eq!(x.grad_vec(), Some(vec![1.; 3]));
        Ok(())
    }

    #[test]
    fn exp() -> Result<(), TensoriaError> {
        let mut x = Tensor::new([3, 2], vec![2., 2., 4., 4., 6., 6.])?;
        x.set_requires_grad(true);

        let res = x.exp();
        assert_eq!(
            res.to_vec(),
            x.to_vec().iter().map(|v| v.exp()).collect::<Vec<f32>>()
        );
        res.backward()?;

        assert_eq!(
            x.grad_vec().unwrap(),
            x.to_vec().iter().map(|v| v.exp()).collect::<Vec<f32>>()
        );
        Ok(())
    }
}
