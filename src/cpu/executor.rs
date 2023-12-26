use std::collections::HashMap;
use std::error::Error;
use std::ops::{Add, Mul};
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

#[derive(Debug)]
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
}

pub struct CPUExecutor {
    tensors: HashMap<Uuid, CPUTensor>,
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
                    if !self.tensors.contains_key(id) {
                        self.tensors.insert(*id, CPUTensor::from_var(&var));
                    }
                }
                VarType::Add => {
                    self.tensors.insert(
                        *id,
                        self.tensors[&var_prevs[0]].add(&self.tensors[&var_prevs[1]]),
                    );
                }
                VarType::Sub => {
                    self.tensors.insert(
                        *id,
                        self.tensors[&var_prevs[0]].sub(&self.tensors[&var_prevs[1]]),
                    );
                }
                VarType::Mul => {
                    self.tensors.insert(
                        *id,
                        self.tensors[&var_prevs[0]].mul(&self.tensors[&var_prevs[1]]),
                    );
                }
                VarType::MatMul => {
                    self.tensors.insert(
                        *id,
                        self.tensors[&var_prevs[0]].matmul(&self.tensors[&var_prevs[1]]),
                    );
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

        if let Some(grad) = self.tensors.get_mut(&var.id) {
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
                VarType::Mul => self.binop_backward(
                    var,
                    |l_g, out_g, _, r_val| Some(l_g.add(r_val.mul(out_g))),
                    |r_g, out_g, l_val, _| Some(r_g.add(l_val.mul(out_g))),
                ),
                _ => todo!(),
            }
        }
        Ok(())
    }
}

type CalcGradFn = fn(
    old_grad: &ArrayD<f32>,
    out_grad: &ArrayD<f32>,
    left_val: &ArrayD<f32>,
    right_val: &ArrayD<f32>,
) -> Option<ArrayD<f32>>;

impl CPUExecutor {
    fn binop_backward(
        &mut self,
        var: &Arc<Variable>,
        calc_left_grad_fn: CalcGradFn,
        calc_right_grad_fn: CalcGradFn,
    ) {
        let var_prevs = var.prevs.clone();
        let id = var.id;

        let new_left_grad = match (
            &self.tensors.get(&var_prevs[0]).unwrap().grad,
            &self.tensors.get(&id).unwrap().grad,
        ) {
            (Some(left_grad), Some(out_grad)) => {
                let mut out_grad = out_grad.clone();
                let left_val = &self.tensors[&var_prevs[0]].data;
                let right_val = &self.tensors[&var_prevs[1]].data;

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
        if let Some(left_tensor) = self.tensors.get_mut(&var_prevs[0]) {
            left_tensor.grad = new_left_grad;
        }

        let new_right_grad = match (
            &self.tensors.get(&var_prevs[1]).unwrap().grad,
            &self.tensors.get(&id).unwrap().grad,
        ) {
            (Some(right_grad), Some(out_grad)) => {
                let mut out_grad = out_grad.clone();
                let left_val = &self.tensors[&var_prevs[0]].data;
                let right_val = &self.tensors[&var_prevs[1]].data;

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
        if let Some(right_tensor) = self.tensors.get_mut(&var_prevs[1]) {
            right_tensor.grad = new_right_grad;
        }
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

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

        let mut executor = CPUExecutor {
            tensors: HashMap::new(),
        };
        executor.forward(&sess).unwrap();
        let res_cpu_add = executor.tensors.get(&res_add.id).unwrap();
        let res_cpu_sub = executor.tensors.get(&res_sub.id).unwrap();
        let res_cpu_mul = executor.tensors.get(&res_mul.id).unwrap();

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

        let mut executor = CPUExecutor {
            tensors: HashMap::new(),
        };
        executor.forward(&sess).unwrap();
        executor.backward(&c, &sess).unwrap();
        if let Some(b_grad) = executor.tensors.get(&b.id).unwrap().grad.as_ref() {
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

        let mut executor = CPUExecutor {
            tensors: HashMap::new(),
        };
        executor.forward(&sess).unwrap();
        executor.backward(&b, &sess).unwrap();
        if let Some(a_grad) = executor.tensors.get(&a.id).unwrap().grad.as_ref() {
            assert_eq!(
                a_grad.as_standard_layout().as_slice().unwrap(),
                vec![3., 3., 3., 3.]
            );
        } else {
            panic!("a_grad should not be None")
        }
    }

    #[test]
    fn add_self() {
        let sess = Session::new();
        let a = sess
            .init_tensor_var(TensorData::F32(vec![1.0, 2.0]), vec![2])
            .unwrap();
        let b = a.add(&a).add(&a);

        let mut executor = CPUExecutor {
            tensors: HashMap::new(),
        };
        executor.forward(&sess).unwrap();
        let res_cpu = executor.tensors.get(&b.id).unwrap();
        assert_eq!(res_cpu.data.as_slice().unwrap(), vec![3.0, 6.0]);
    }

    #[test]
    fn mul_backward() {
        let sess = Session::new();
        let a = sess
            .init_tensor_var_with_grad(TensorData::F32(vec![1.0, 2.0]), vec![2])
            .unwrap();
        let b = a.mul(&a);

        let mut executor = CPUExecutor {
            tensors: HashMap::new(),
        };
        executor.forward(&sess).unwrap();
        executor.backward(&b, &sess).unwrap();

        if let Some(a_grad) = executor.tensors.get(&a.id).unwrap().grad.as_ref() {
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

        let mut executor = CPUExecutor {
            tensors: HashMap::new(),
        };

        executor.forward(&sess).unwrap();
        executor.backward(&c, &sess).unwrap();

        if let Some(a_grad) = executor.tensors.get(&a.id).unwrap().grad.as_ref() {
            assert_eq!(
                a_grad.as_standard_layout().as_slice().unwrap(),
                vec![1., 2., 1., 2.]
            );
        } else {
            panic!("a_grad should not be None")
        }
        if let Some(b_grad) = executor.tensors.get(&b.id).unwrap().grad.as_ref() {
            assert_eq!(
                b_grad.as_standard_layout().as_slice().unwrap(),
                vec![4., 6.]
            );
        } else {
            panic!("b_grad should not be None")
        }
    }
}
