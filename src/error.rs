use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum TensoriaError {
    CannotReshapeError,
    AccessingMismatchedType,
    DeviceNotCreated,
    BackwardOnTensorWithNoGrad,
    AlreadyGPUTensor,
}

impl Display for TensoriaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl std::error::Error for TensoriaError {}
