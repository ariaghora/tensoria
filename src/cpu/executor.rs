use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;
use std::sync::Arc;

use ndarray::{ArrayD, Axis, Ix2};
use uuid::Uuid;

use crate::error;
use crate::session::Session;
use crate::traits::{Executor, TensorProps};
use crate::var::{TensorDataType, Variable, VarType};

#[derive(Debug)]
pub enum CPUTensorData {
    F32(ndarray::ArrayD<f32>),
    // I32(ndarray::ArrayD<i32>),
}

impl TensorProps for CPUTensorData {
    fn shape(&self) -> Vec<usize> {
        match self {
            CPUTensorData::F32(val) => val.shape().to_vec(),
        }
    }

    fn dtype(&self) -> TensorDataType {
        match self {
            CPUTensorData::F32(..) => TensorDataType::F32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CPUTensor {
    pub data: ArrayD<f32>,
    pub grad: Option<ArrayD<f32>>,
    pub requires_grad: bool,
}

impl CPUTensor {
    pub fn from_var(var: &Arc<Variable>) -> Self {
        match &var.tensor_data {
            Some(data) => match data.dtype() {
                TensorDataType::F32 => CPUTensor::new(
                    ArrayD::from_shape_vec(var.shape.as_slice(), data.get_data_f32().clone())
                        .unwrap()
                        .into_dyn(),
                    var.requires_grad,
                ),
                _ => panic!("Should be f32 :("),
            },
            None => panic!("Cannot build CPUTensor from None"),
        }
    }

    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> CPUTensor {
        let zero_grad = |shape, requires_grad: bool| {
            if requires_grad {
                Some(ArrayD::zeros(shape))
            } else {
                None
            }
        };

        CPUTensor {
            data: data.clone(),
            grad: zero_grad(data.shape(), requires_grad),
            requires_grad,
        }
    }
}

impl CPUTensor {
    pub fn add(&self, other: &CPUTensor) -> CPUTensor {
        let requires_grad = self.requires_grad || other.requires_grad;
        CPUTensor::new(&self.data + &other.data, requires_grad)
    }

    pub fn sub(&self, other: &CPUTensor) -> CPUTensor {
        let requires_grad = self.requires_grad || other.requires_grad;
        CPUTensor::new(&self.data - &other.data, requires_grad)
    }

    pub fn matmul(&self, other: &CPUTensor) -> CPUTensor {
        let requires_grad = self.requires_grad || other.requires_grad;
        let left = &self.data.clone().into_dimensionality::<Ix2>().unwrap();
        let right = &other.data.clone().into_dimensionality::<Ix2>().unwrap();
        let res = left.dot(right).into_dyn();
        CPUTensor::new(res, requires_grad)
    }

    pub fn mul(&self, other: &CPUTensor) -> CPUTensor {
        let requires_grad = self.requires_grad || other.requires_grad;
        CPUTensor::new(&self.data * &other.data, requires_grad)
    }

    pub fn mean(&self) -> CPUTensor {
        let mu_val = self.data.mean().unwrap();
        CPUTensor::new(
            ArrayD::from_shape_vec(vec![1].as_slice(), vec![mu_val])
                .unwrap()
                .into_dyn(),
            self.requires_grad,
        )
    }
}

pub struct CPUExecutor {
    pub tensors: Rc<RefCell<HashMap<Uuid, CPUTensor>>>,
    pub staging_tensors: Rc<RefCell<HashMap<Uuid, CPUTensor>>>,
}

impl Executor for CPUExecutor {
    fn forward(&mut self, session: &Session) -> Result<(), Box<dyn Error>> {
        let sorted_id = session.sorted_ids();
        for id in &sorted_id {
            let var = &session.variables.borrow()[id];
            let var_type = var.var_type.clone();
            let var_prevs = var.prevs.clone();
            match var_type {
                VarType::Leaf => {
                    if self.staging_tensors.borrow().contains_key(id) {
                        self.tensors.borrow_mut().insert(*id, self.staging_tensors.borrow()[id].clone());
                    } else {
                        self.tensors
                            .borrow_mut()
                            .insert(*id, CPUTensor::from_var(&var));
                    }
                }
                VarType::Add => {
                    let t = self.tensors.borrow()[&var_prevs[0]]
                        .add(&self.tensors.borrow()[&var_prevs[1]]);
                    self.tensors.borrow_mut().insert(*id, t);
                }
                VarType::Sub => {
                    let t = self.tensors.borrow()[&var_prevs[0]]
                        .sub(&self.tensors.borrow()[&var_prevs[1]]);
                    self.tensors.borrow_mut().insert(*id, t);
                }
                VarType::Mul => {
                    let t = self.tensors.borrow()[&var_prevs[0]]
                        .mul(&self.tensors.borrow()[&var_prevs[1]]);
                    self.tensors.borrow_mut().insert(*id, t);
                }
                VarType::MatMul => {
                    let t = self.tensors.borrow()[&var_prevs[0]]
                        .matmul(&self.tensors.borrow()[&var_prevs[1]]);
                    self.tensors.borrow_mut().insert(*id, t);
                }
                VarType::Mean => {
                    let t = self.tensors.borrow()[&var_prevs[0]].mean();
                    self.tensors.borrow_mut().insert(*id, t);
                }
            };
        }
        Ok(())
    }

    fn backward(&mut self, var: &Arc<Variable>, session: &Session) -> Result<(), Box<dyn Error>> {
        // Set output grad
        if !var.requires_grad {
            return Err(Box::new(error::TensoriaError::BackwardOnTensorWithNoGrad));
        }

        if let Some(grad) = self.tensors.borrow_mut().get_mut(&var.id) {
            grad.grad = Some(ArrayD::ones(var.shape.as_slice()));
        }

        let sorted_id: Vec<Uuid> = session.sorted_ids().into_iter().rev().collect();
        for id in &sorted_id {
            let var = &session.variables.borrow()[&id];

            match var.var_type {
                VarType::Leaf => continue,
                VarType::Add => self.binop_backward(
                    var,
                    |l_g, out_g, _, _| Some(l_g.add(out_g)),
                    |r_g, out_g, _, _| Some(r_g.add(out_g)),
                ),
                VarType::Sub => self.binop_backward(
                    var,
                    |l_g, out_g, _, _| Some(l_g.add(out_g)),
                    |r_g, out_g, _, _| Some(r_g.sub(out_g)),
                ),
                VarType::MatMul => self.binop_backward(
                    var,
                    |l_g, out_g, _, r_val| {
                        let out_g_mat = out_g.clone().into_dimensionality::<Ix2>().unwrap();
                        let r_val_mat = r_val.clone().into_dimensionality::<Ix2>().unwrap();
                        Some(l_g.add(out_g_mat.dot(&r_val_mat.t())))
                    },
                    |r_g, out_g, l_val, _| {
                        let out_g_mat = out_g.clone().into_dimensionality::<Ix2>().unwrap();
                        let l_val_mat = l_val.clone().into_dimensionality::<Ix2>().unwrap();
                        Some(r_g.add(l_val_mat.t().dot(&out_g_mat)))
                    },
                ),
                VarType::Mean => self.unop_backward(var, |var_g, out_g, var_val| {
                    let out_g_mean = out_g / var_val.len() as f32;
                    let ones = ArrayD::from_elem(var_val.shape(), 1.0);
                    Some(var_g.add(out_g_mean.mul(&ones)))
                }),
                VarType::Mul => self.binop_backward(
                    var,
                    |l_g, out_g, _, r_val| Some(l_g.add(r_val.mul(out_g))),
                    |r_g, out_g, l_val, _| Some(r_g.add(l_val.mul(out_g))),
                ),
                _ => todo!(),
            }
        }

        // Move tensors from the last forward propagation into the staging map, then reset the
        // main working tensor map
        self.staging_tensors = self.tensors.clone();
        self.tensors = Default::default();

        // Clear intermediary variables and reset leaf variables' outgoing edges
        for id in &session.intermediary_ids() {
            session.variables.borrow_mut().remove(&id);
        }
        for (_, v) in session.variables.borrow_mut().iter() {
            v.nexts.borrow_mut().clear();
        }

        Ok(())
    }
}

type CalcBinopGradFn = fn(
    old_grad: &ArrayD<f32>,
    out_grad: &ArrayD<f32>,
    left_val: &ArrayD<f32>,
    right_val: &ArrayD<f32>,
) -> Option<ArrayD<f32>>;

type CalcUnopGradFn =
fn(old_grad: &ArrayD<f32>, out_grad: &ArrayD<f32>, val: &ArrayD<f32>) -> Option<ArrayD<f32>>;

type UpdateFn<T> = fn(val: &T, grad: &T) -> T;

impl CPUExecutor {
    pub fn new() -> Self {
        Self {
            tensors: Default::default(),
            staging_tensors: Default::default(),
        }
    }

    pub fn step(&self, var: &Arc<Variable>, update_fn: UpdateFn<ArrayD<f32>>) {
        let new_val_opt = {
            let tensors = self.staging_tensors.borrow();
            let t = tensors.get(&var.id).unwrap();
            let val = &t.data;
            if let Some(grad) = &t.grad {
                let new_val = update_fn(val, grad);
                Some(new_val)
            } else { None }
        };

        let mut tensors_mut = self.staging_tensors.borrow_mut();
        if let (Some(t), Some(new_val)) = (tensors_mut.get_mut(&var.id), new_val_opt) {
            t.data = new_val;
            t.grad = Some(ArrayD::from_elem(var.shape.clone(), 0.0).into());
        }
    }

    pub fn fetch(&self, var: &Arc<Variable>) -> Option<CPUTensor> {
        // Try to lookup at the staging map. If not found, then probably it is the first pass without
        // backward pass being called and try to lookup at main working map. Otherwise, simply return
        // None.
        if let Some(val) = self.staging_tensors.borrow().get(&var.id) {
            return Some(val.clone());
        } else if let Some(val) = self.tensors.borrow().get(&var.id) {
            return Some(val.clone());
        } else {
            None
        }
    }

    fn binop_backward(
        &mut self,
        var: &Arc<Variable>,
        calc_left_grad_fn: CalcBinopGradFn,
        calc_right_grad_fn: CalcBinopGradFn,
    ) {
        let var_prevs = var.prevs.clone();
        let id = var.id;

        let new_left_grad = match (
            &self.tensors.borrow().get(&var_prevs[0]).unwrap().grad,
            &self.tensors.borrow().get(&id).unwrap().grad,
        ) {
            (Some(left_grad), Some(out_grad)) => {
                let mut out_grad = out_grad.clone();
                let left_val = &self.tensors.borrow()[&var_prevs[0]].data;
                let right_val = &self.tensors.borrow()[&var_prevs[1]].data;

                // Sum out added dims
                let ndims_added = out_grad.ndim() - left_val.ndim();
                for _ in 0..ndims_added {
                    out_grad = out_grad.sum_axis(Axis(0));
                }

                // Sum across broadcasted but non-added dims
                for (i, dim) in left_val.shape().iter().enumerate() {
                    if dim == &1 {
                        out_grad = out_grad.sum_axis(Axis(i)).insert_axis(Axis(i));
                    }
                }

                let new_left_grad = calc_left_grad_fn(left_grad, &out_grad, left_val, right_val);
                new_left_grad
            }
            _ => None,
        };
        if let Some(left_tensor) = self.tensors.borrow_mut().get_mut(&var_prevs[0]) {
            left_tensor.grad = new_left_grad;
        }

        let new_right_grad = match (
            &self.tensors.borrow().get(&var_prevs[1]).unwrap().grad,
            &self.tensors.borrow().get(&id).unwrap().grad,
        ) {
            (Some(right_grad), Some(out_grad)) => {
                let mut out_grad = out_grad.clone();
                let left_val = &self.tensors.borrow()[&var_prevs[0]].data;
                let right_val = &self.tensors.borrow()[&var_prevs[1]].data;

                // Sum out added dims
                let ndims_added = out_grad.ndim() - right_val.ndim();
                for _ in 0..ndims_added {
                    out_grad = out_grad.sum_axis(Axis(0));
                }

                // Sum across broadcasted but non-added dims
                for (i, dim) in right_val.shape().iter().enumerate() {
                    if dim == &1 {
                        out_grad = out_grad.sum_axis(Axis(i)).insert_axis(Axis(i));
                    }
                }
                let new_left_grad = calc_right_grad_fn(right_grad, &out_grad, left_val, right_val);
                new_left_grad
            }
            _ => None,
        };
        if let Some(right_tensor) = self.tensors.borrow_mut().get_mut(&var_prevs[1]) {
            right_tensor.grad = new_right_grad;
        }
    }

    fn unop_backward(&mut self, var: &Arc<Variable>, calc_grad_fn: CalcUnopGradFn) {
        let var_prevs = var.prevs.clone();
        let id = var.id;

        let new_grad = match (
            &self.tensors.borrow().get(&var_prevs[0]).unwrap().grad,
            &self.tensors.borrow().get(&id).unwrap().grad,
        ) {
            (Some(old_grad), Some(out_grad)) => {
                let out_grad = out_grad.clone();
                let val = &self.tensors.borrow()[&var_prevs[0]].data;
                let new_grad = calc_grad_fn(old_grad, &out_grad, val);
                new_grad
            }
            _ => None,
        };
        if let Some(left_tensor) = self.tensors.borrow_mut().get_mut(&var_prevs[0]) {
            left_tensor.grad = new_grad;
        }
    }
}

#[cfg(test)]
mod test {
    use crate::cpu::executor::CPUExecutor;
    use crate::session::Session;
    use crate::traits::Executor;
    use crate::var::TensorData;

    #[test]
    fn basic_op() {
        let sess = Session::new();
        let a = sess
            .init_tensor_var(TensorData::F32(vec![1.0, 2.0]), vec![2])
            .unwrap();
        let b = sess
            .init_tensor_var(TensorData::F32(vec![3.0, 4.0]), vec![2])
            .unwrap();

        let res_add = a.add(&b);
        let res_sub = a.sub(&b);
        let res_mul = a.mul(&b);

        let mut executor = CPUExecutor::new();
        executor.forward(&sess).unwrap();
        let res_cpu_add = executor.fetch(&res_add).unwrap();
        let res_cpu_sub = executor.fetch(&res_sub).unwrap();
        let res_cpu_mul = executor.fetch(&res_mul).unwrap();

        {
            assert_eq!(res_cpu_add.data.as_slice().unwrap(), vec![4.0, 6.0]);
            assert_eq!(res_cpu_sub.data.as_slice().unwrap(), vec![-2.0, -2.0]);
            assert_eq!(res_cpu_mul.data.as_slice().unwrap(), vec![3.0, 8.0]);
        }
    }

    #[test]
    fn bin_op_bcast() {
        let sess = Session::new();
        let a = sess
            .init_tensor_var_with_grad(TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2])
            .unwrap();
        let b = sess
            .init_tensor_var_with_grad(TensorData::F32(vec![1.0, 2.0]), vec![2, 1])
            .unwrap();
        let c = a.add(&b);

        let mut executor = CPUExecutor::new();
        executor.forward(&sess).unwrap();
        executor.backward(&c, &sess).unwrap();
        if let Some(b_grad) = executor.fetch(&b).unwrap().grad.as_ref() {
            assert_eq!(
                b_grad.as_standard_layout().as_slice().unwrap(),
                vec![2., 2.]
            );
        } else {
            panic!("b_grad should not be None")
        }
    }

    #[test]
    fn add_backward() {
        let sess = Session::new();
        let a = sess
            .init_tensor_var_with_grad(TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2])
            .unwrap();
        let b = a.add(&a).add(&a);

        let mut executor = CPUExecutor::new();
        executor.forward(&sess).unwrap();
        executor.backward(&b, &sess).unwrap();
        if let Some(a_grad) = executor.fetch(&a).unwrap().grad.as_ref() {
            assert_eq!(
                a_grad.as_standard_layout().as_slice().unwrap(),
                vec![3., 3., 3., 3.]
            );
        } else {
            panic!("a_grad should not be None");
        }
    }

    #[test]
    fn add_self() {
        let sess = Session::new();
        let a = sess
            .init_tensor_var(TensorData::F32(vec![1.0, 2.0]), vec![2])
            .unwrap();
        let b = a.add(&a).add(&a);

        let mut executor = CPUExecutor::new();
        executor.forward(&sess).unwrap();
        let res_cpu = executor.fetch(&b).unwrap();
        assert_eq!(res_cpu.data.as_slice().unwrap(), vec![3.0, 6.0]);
    }

    #[test]
    fn mul_backward() {
        let sess = Session::new();
        let a = sess
            .init_tensor_var_with_grad(TensorData::F32(vec![1.0, 2.0]), vec![2])
            .unwrap();
        let b = a.mul(&a);

        let mut executor = CPUExecutor::new();
        executor.forward(&sess).unwrap();
        executor.backward(&b, &sess).unwrap();

        if let Some(a_grad) = executor.fetch(&a).unwrap().grad.as_ref() {
            assert_eq!(a_grad.as_slice().unwrap(), vec![2.0, 4.0]);
        } else {
            panic!("a_grad should not be None")
        }
    }

    #[test]
    fn matmul() {
        let sess = Session::new();
        let a = sess
            .init_tensor_var_with_grad(TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2])
            .unwrap();
        let b = sess
            .init_tensor_var_with_grad(TensorData::F32(vec![1.0, 2.0]), vec![2, 1])
            .unwrap();
        let c = a.matmul(&b);

        let mut executor = CPUExecutor::new();

        executor.forward(&sess).unwrap();
        executor.backward(&c, &sess).unwrap();

        if let Some(a_grad) = executor.fetch(&a).unwrap().grad.as_ref() {
            assert_eq!(
                a_grad.as_standard_layout().as_slice().unwrap(),
                vec![1., 2., 1., 2.]
            );
        } else {
            panic!("a_grad should not be None")
        }
        if let Some(b_grad) = executor.fetch(&b).unwrap().grad.as_ref() {
            assert_eq!(
                b_grad.as_standard_layout().as_slice().unwrap(),
                vec![4., 6.]
            );
        } else {
            panic!("b_grad should not be None")
        }
    }
}
