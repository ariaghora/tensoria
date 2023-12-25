use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use ndarray::{ArrayD, IxDyn};
use uuid::Uuid;

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
            CPUTensorData::F32(val) => { val.shape().to_vec() }
        }
    }

    fn dtype(&self) -> TensorDataType {
        match self {
            CPUTensorData::F32(..) => { TensorDataType::F32 }
        }
    }
}

#[derive(Debug)]
pub struct CPUTensor {
    pub data: CPUTensorData,
    pub grad: Option<CPUTensorData>,
    pub requires_grad: bool,
}

impl CPUTensor {
    pub fn from_var(var: &Arc<Variable>) -> Self {
        match &var.tensor_data {
            Some(data) => {
                match data.dtype() {
                    TensorDataType::F32 => {
                        CPUTensor {
                            data: CPUTensorData::F32(
                                ArrayD::from_shape_vec(IxDyn(var.shape.as_slice()), data.get_data_f32().clone()).unwrap().into_dyn()
                            ),
                            grad: None,
                            requires_grad: false,
                        }
                    }
                    _ => panic!("Should be f32 :(")
                }
            }
            None => panic!("Cannot build CPUTensor from None")
        }
    }

    pub fn new(data: CPUTensorData, requires_grad: bool) -> CPUTensor {
        let zero_grad = |shape, requires_grad: bool| {
            if requires_grad {
                Some(CPUTensorData::F32(ArrayD::zeros(shape)))
            } else {
                None
            }
        };

        // Function to handle addition and return a CPUTensor
        let add_tensors = |data: CPUTensorData, requires_grad: bool| {
            let shape = data.shape().clone();
            CPUTensor {
                data,
                grad: zero_grad(shape, requires_grad),
                requires_grad,
            }
        };
        add_tensors(data, requires_grad)
    }
}

impl CPUTensor {
    pub fn add(&self, other: &CPUTensor) -> CPUTensor {
        let requires_grad = self.requires_grad || other.requires_grad;
        match (&self.data, &other.data) {
            (CPUTensorData::F32(a), CPUTensorData::F32(b)) => {
                CPUTensor::new(CPUTensorData::F32(a + b), requires_grad)
            }
            _ => unimplemented!(),
        }
    }

    pub fn sub(&self, other: &CPUTensor) -> CPUTensor {
        let requires_grad = self.requires_grad || other.requires_grad;
        match (&self.data, &other.data) {
            (CPUTensorData::F32(a), CPUTensorData::F32(b)) => {
                CPUTensor::new(CPUTensorData::F32(a - b), requires_grad)
            }
            _ => unimplemented!(),
        }
    }

    pub fn mul(&self, other: &CPUTensor) -> CPUTensor {
        let requires_grad = self.requires_grad || other.requires_grad;
        match (&self.data, &other.data) {
            (CPUTensorData::F32(a), CPUTensorData::F32(b)) => {
                CPUTensor::new(CPUTensorData::F32(a * b), requires_grad)
            }
            _ => unimplemented!(),
        }
    }
}

pub struct CPUExecutor {
    tensors: HashMap<Uuid, CPUTensor>,
}

impl Executor for CPUExecutor {
    fn forward(&mut self, session: &Session) -> Result<(), Box<dyn Error>> {
        let sorted_id = session.sorted_ids();
        for id in &sorted_id {
            let var = &session.variables.borrow()[id].clone();
            let var_type = var.var_type.clone();
            let var_prevs = var.prevs.clone();
            match var_type {
                VarType::Leaf => { self.tensors.insert(*id, CPUTensor::from_var(&var)); }
                VarType::Add => { self.tensors.insert(*id, self.tensors[&var_prevs[0]].add(&self.tensors[&var_prevs[1]])); }
                VarType::Sub => { self.tensors.insert(*id, self.tensors[&var_prevs[0]].sub(&self.tensors[&var_prevs[1]])); }
                VarType::Mul => { self.tensors.insert(*id, self.tensors[&var_prevs[0]].mul(&self.tensors[&var_prevs[1]])); }
                _ => todo!()
            };
        }
        Ok(())
    }

    fn backward(&self, var: &Arc<Variable>, session: &Session) -> Result<(), Box<dyn Error>> {
        todo!()
    }
}

impl CPUExecutor {
    fn binop_forward(&mut self, id: Uuid, var: &Arc<Variable>, output_tensor: CPUTensor) {
        self.tensors.insert(id, output_tensor);
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use crate::cpu::executor::{CPUExecutor, CPUTensorData};
    use crate::session::Session;
    use crate::traits::Executor;
    use crate::var::TensorData;

    #[test]
    fn basic_op() {
        let sess = Session::new();
        let a = sess.init_tensor_var(TensorData::F32(vec![1.0, 2.0]), vec![2]).unwrap();
        let b = sess.init_tensor_var(TensorData::F32(vec![3.0, 4.0]), vec![2]).unwrap();

        let res_add = a.add(&b);
        let res_sub = a.sub(&b);
        let res_mul = a.mul(&b);

        let mut executor = CPUExecutor { tensors: HashMap::new() };
        executor.forward(&sess).unwrap();
        let res_cpu_add = executor.tensors.get(&res_add.id).unwrap();
        let res_cpu_sub = executor.tensors.get(&res_sub.id).unwrap();
        let res_cpu_mul = executor.tensors.get(&res_mul.id).unwrap();

        if let (
            CPUTensorData::F32(data_add),
            CPUTensorData::F32(data_sub),
            CPUTensorData::F32(data_mul),
        ) = (&res_cpu_add.data, &res_cpu_sub.data, &res_cpu_mul.data) {
            assert_eq!(data_add.as_slice().unwrap(), vec![4.0, 6.0]);
            assert_eq!(data_sub.as_slice().unwrap(), vec![-2.0, -2.0]);
            assert_eq!(data_mul.as_slice().unwrap(), vec![3.0, 8.0]);
        } else {
            panic!("result should be of type F32")
        }
    }

    #[test]
    fn add_self() {
        let sess = Session::new();
        let a = sess.init_tensor_var(TensorData::F32(vec![1.0, 2.0]), vec![2]).unwrap();
        let b = a.add(&a).add(&a);

        let mut executor = CPUExecutor { tensors: HashMap::new() };
        executor.forward(&sess).unwrap();
        let res_cpu = executor.tensors.get(&b.id).unwrap();
        if let CPUTensorData::F32(data) = &res_cpu.data {
            assert_eq!(data.as_slice().unwrap(), vec![3.0, 6.0])
        } else {
            panic!("result should be of type F32")
        }
    }
}