use std::sync::Arc;

use rand::distributions::{Distribution, Uniform};

use crate::session::Session;
use crate::var::{TensorData, Variable};

pub trait Forward {
    fn forward(&self, x: &Arc<Variable>) -> Arc<Variable>;
}

pub struct Linear {
    w: Arc<Variable>,
    b: Arc<Variable>,
}

impl Forward for Linear {
    fn forward(&self, x: &Arc<Variable>) -> Arc<Variable> {
        x.matmul(&self.w).add(&self.b)
    }
}

impl Linear {
    pub fn new(session: &Session, in_size: usize, out_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let xavier_limit = (6.0f32 / (in_size + out_size) as f32).sqrt();
        let uniform = Uniform::new(-xavier_limit, xavier_limit);

        let w_val: Vec<f32> = (0..in_size * out_size)
            .map(|_| uniform.sample(&mut rng))
            .collect();
        let b_val: Vec<f32> = vec![0.0; out_size]; // Biases can be initialized to zero
        Self {
            w: session
                .init_tensor_var_with_grad(TensorData::F32(w_val), vec![in_size, out_size])
                .unwrap(),
            b: session
                .init_tensor_var_with_grad(TensorData::F32(b_val), vec![out_size])
                .unwrap(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::cpu::executor::CPUExecutor;
    use crate::nn::{Forward, Linear};
    use crate::session::Session;
    use crate::traits::Executor;
    use crate::var::TensorData;

    #[test]
    fn test_linear() {
        let sess = Session::new();
        let x = sess
            .init_tensor_var(
                TensorData::F32(vec![1., 2., 3., 4., 1., 2., 3., 4.]),
                vec![2, 4],
            )
            .unwrap();

        let l1 = Linear::new(&sess, 4, 2);
        let y = l1.forward(&x);

        let mut exec = CPUExecutor::new();
        exec.forward(&sess).unwrap();

        assert!(y.requires_grad);
        if let Some(t) = exec.fetch(&y) {
            assert_eq!(t.data.shape(), &[2, 2]);
        } else {
            panic!("Cannot be None")
        }
    }
}
