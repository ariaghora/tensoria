pub mod cpu;
pub mod session;
pub mod var;
pub mod error;
pub mod traits;
pub mod wgpu;


#[cfg(test)]
mod tests {
    use crate::session::Session;
    use crate::var::TensorData;

    #[test]
    fn with_shape_should_succeed() {
        let sess = Session::new();
        assert!(sess.new_tensor_var(TensorData::I32(vec![2]), vec![0]).is_ok());
        assert!(sess.new_tensor_var(TensorData::F32(vec![2.1]), vec![1]).is_ok());
        assert!(sess.new_tensor_var(TensorData::I32(vec![2, 2]), vec![2]).is_ok());
        assert!(sess.new_tensor_var(TensorData::I32(vec![2, 2]), vec![2, 1]).is_ok());
    }

    #[test]
    fn with_shape_should_fail() {
        let sess = Session::new();
        assert!(sess.new_tensor_var(TensorData::I32(vec![1, 2]), vec![3]).is_err());
    }

    #[test]
    fn dependency() {
        let sess = Session::new();
        let a = sess.new_tensor_var(TensorData::F32(vec![1.0]), vec![]).unwrap();
        let b = sess.new_tensor_var(TensorData::F32(vec![1.0]), vec![]).unwrap();
        let c = a.add(&b);
        assert_eq!(c.prevs[0], a.id);
        assert_eq!(c.prevs[1], b.id);
    }

    #[test]
    fn recursive_dependency() {
        let sess = Session::new();
        let a = sess.new_tensor_var(TensorData::F32(vec![1.0]), vec![]).unwrap();
        let b = a.add(&a);
        assert_eq!(b.prevs[0], a.id);
        assert_eq!(b.prevs[1], a.id);
        assert_eq!(sess.tensors.borrow().len(), 2);
    }
}
