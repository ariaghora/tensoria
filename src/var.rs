use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use uuid::Uuid;

#[derive(Debug, Clone, PartialEq)]
pub enum VarType {
    Add,
    Sub,
    Leaf,
    MatMul,
    Mul,
}

#[derive(Debug, Clone)]
pub enum TensorDataType {
    F32,
    I32,
}

impl TensorDataType {
    pub fn wgsl_type(&self) -> &str {
        match self {
            TensorDataType::F32 => "f32",
            TensorDataType::I32 => "i32",
        }
    }
}

#[derive(Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl TensorData {
    pub(crate) fn len(&self) -> usize {
        match self {
            TensorData::F32(val) => val.len(),
            TensorData::I32(val) => val.len(),
        }
    }

    pub fn dtype(&self) -> TensorDataType {
        match self {
            TensorData::F32(_) => TensorDataType::F32,
            TensorData::I32(_) => TensorDataType::I32,
        }
    }

    pub fn get_data_f32<'a, 'b>(&'a self) -> &'b Vec<f32>
    where
        'a: 'b,
    {
        if let TensorData::F32(val) = self {
            return val;
        }
        panic!("Attempting to get f32 but the tensor is of different type")
    }
    pub fn get_data_i32<'a, 'b>(&'a self) -> &'b Vec<i32>
    where
        'a: 'b,
    {
        if let TensorData::I32(val) = self {
            return val;
        }
        panic!("Attempting to get f32 but the tensor is of different type")
    }
}

#[derive(Debug)]
pub struct Variable {
    pub id: Uuid,
    pub tensor_data: Option<TensorData>,
    pub dtype: TensorDataType,
    pub shape: Vec<usize>,
    pub session: Weak<RefCell<HashMap<Uuid, Arc<Variable>>>>,
    pub var_type: VarType,
    pub prevs: Vec<Uuid>,
    pub nexts: Rc<RefCell<Vec<Uuid>>>,
    pub requires_grad: bool,
}

impl Variable {
    fn binary_op(
        &self,
        other: &Arc<Variable>,
        var_type: VarType,
        output_shape: Vec<usize>,
    ) -> Arc<Variable> {
        let output_dtype = match (&self.dtype, &other.dtype) {
            (TensorDataType::F32, TensorDataType::F32) => TensorDataType::F32,
            _ => unimplemented!(
                "cannot perform {:?} between {:?} and {:?}",
                var_type,
                &self.dtype,
                &other.dtype
            ),
        };

        let result_tensor = Arc::new(Variable {
            id: Uuid::new_v4(),
            tensor_data: None,
            dtype: output_dtype,
            shape: output_shape,
            session: self.session.clone(),
            prevs: vec![self.id, other.id],
            nexts: Rc::new(RefCell::new(Vec::new())),
            var_type,
            requires_grad: self.requires_grad || other.requires_grad,
        });

        self.nexts.borrow_mut().push(self.id);
        other.nexts.borrow_mut().push(self.id);

        if let Some(session) = self.session.upgrade() {
            session
                .borrow_mut()
                .insert(result_tensor.id, result_tensor.clone());
        }
        result_tensor
    }

    pub fn add(&self, other: &Arc<Variable>) -> Arc<Variable> {
        self.binary_op(other, VarType::Add, self.shape.clone())
    }

    pub fn sub(&self, other: &Arc<Variable>) -> Arc<Variable> {
        self.binary_op(other, VarType::Sub, self.shape.clone())
    }

    pub fn matmul(&self, other: &Arc<Variable>) -> Arc<Variable> {
        // TODO: provide a separate shape validation for all vars
        let shape = vec![self.shape[0], other.shape[1]];
        self.binary_op(other, VarType::MatMul, shape)
    }

    pub fn mul(&self, other: &Arc<Variable>) -> Arc<Variable> {
        self.binary_op(other, VarType::Mul, self.shape.clone())
    }
}

#[cfg(test)]
mod test {
    use crate::session::Session;
    use crate::var::TensorData;

    #[test]
    fn links() {
        let sess = Session::new();
        let a = sess
            .init_tensor_var(TensorData::F32(vec![1.0]), vec![])
            .unwrap();
        let b = sess
            .init_tensor_var(TensorData::F32(vec![1.0]), vec![])
            .unwrap();
        let res = a.add(&b);

        assert_eq!(a.nexts.borrow().len(), 1);
        assert_eq!(b.nexts.borrow().len(), 1);
        assert_eq!(res.nexts.borrow().len(), 0);
        assert_eq!(res.prevs.len(), 2);
    }
}
