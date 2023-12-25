use uuid::Uuid;

use crate::session::Session;
use crate::wgpu::op::Op;

pub struct OpMul {}

impl Op for OpMul {
    fn setup_shader_forward(&self, id: Uuid, session: &Session, params: &mut tera::Context) {
        let op = &session.variables.borrow()[&id];
        let left = &session.variables.borrow()[&op.prevs[0]];
        let right = &session.variables.borrow()[&op.prevs[1]];
        params.insert("input_0_type", left.dtype.wgsl_type());
        params.insert("input_1_type", right.dtype.wgsl_type());
        params.insert("output_0_type", op.dtype.wgsl_type());
    }

    fn setup_shader_backward(&self, id: Uuid, session: &Session, params: &mut tera::Context) {
        let op = &session.variables.borrow()[&id];
        let left = &session.variables.borrow()[&op.prevs[0]];
        let right = &session.variables.borrow()[&op.prevs[1]];
        params.insert("input_0_type", left.dtype.wgsl_type());
        params.insert("grad_0_type", left.dtype.wgsl_type());
        params.insert("input_1_type", right.dtype.wgsl_type());
        params.insert("grad_1_type", right.dtype.wgsl_type());
        params.insert("output_0_grad_type", op.dtype.wgsl_type());

        let left_var = &session.variables.borrow()[&id];
        if left_var.requires_grad {
            params.insert("left_requires_grad", &true);
        }
    }

    fn workgroups(&self, id: Uuid, session: &Session) -> [u32; 3] {
        let local_size_x = 256;

        let out_shape = &session.variables.borrow()[&id].shape;
        let num_elements = out_shape.iter().fold(1, |x, y| x * y);
        let num_workgroups_x = (num_elements + local_size_x - 1) / local_size_x;
        [num_workgroups_x as u32, 1, 1]
    }
}

#[cfg(test)]
mod test {
    use crate::session::Session;
    use crate::traits::Executor;
    use crate::var::TensorData;
    use crate::wgpu::executor::GPUExecutor;

    #[test]
    fn mul() {
        let mut sess = Session::new();
        let a = sess.init_tensor_var(TensorData::F32(vec![1., 2., 3.]), vec![3]).unwrap();
        let b = sess.init_tensor_var(TensorData::F32(vec![1., 2., 3.]), vec![3]).unwrap();
        let c = a.mul(&b);
        let mut executor = GPUExecutor::new();

        executor.forward(&mut sess).unwrap();
        if let TensorData::F32(val) = &executor.fetch(c) {
            assert_eq!(val, &vec![1., 4., 9.])
        } else {
            panic!("Result should be F32")
        }
    }

    #[test]
    fn mul_grad() {
        let mut sess = Session::new();
        let a = sess.init_tensor_var(TensorData::F32(vec![1., 2., 3.]), vec![3]).unwrap();
        let b = sess.init_tensor_var_with_grad(TensorData::F32(vec![1., 2., 3.]), vec![3]).unwrap();
        let c = a.mul(&b);

        assert!(c.requires_grad);
        let mut executor = GPUExecutor::new();

        executor.forward(&mut sess).unwrap();
        // grad tensor data should be created
        assert!(&executor.tensors.borrow()[&c.id].grad.is_some());

        executor.backward(&c, &sess).unwrap();
    }
}
