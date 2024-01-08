<h3 align="center">
    <b>T E N S O R I A</b>
</h3>

<h4 align="center">
    ᕕ(⌐■_■)ᕗ ♪♬
</h4>

---

<p align="center">
An ergonomic tensor manipulation library running on GPU, self-contained, in pure rust
</p>

## Features

- Supports GPU with CPU fallback.
- Provides automatic gradient computation (autograd).
- Allows creation of tensors with arbitrary dimensions at runtime.
- Offers an ergonomic API. **Note:** As a trade-off, tensor operations' shape checking occurs at runtime, potentially
  causing panics due to shape incompatibilities.

## Example

```rust
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let x = Tensor::new([1, 2], vec![1., 2.])?;
    let y = Tensor::new([1, 2], vec![3., 4.])?;
    let res = &x + &y;
    assert_eq!(res.data(), vec![4., 6.]);

    // Or use GPU (via WGPU) if you wish by calling `.to_gpu()`.
    // The tensor will now operate on GPU array, while maintaining
    // the same user-facing API.
    let x = Tensor::new([1, 2], vec![1., 2.])?.to_gpu()?;
    let y = Tensor::new([1, 2], vec![3., 4.])?.to_gpu()?;
    let res = &x + &y;
    assert_eq!(res.data(), vec![4., 6.]);

    // Autograd...
    let mut x = Tensor::new([2, 2], vec![1, 2, 3, 4])?.to_gpu()?;
    x.set_requires_grad(true);

    let res = x.mul(&x).mul(&x);
    res.backward()?;
    assert_eq!(x.grad().unwrap(), vec![3, 12, 27, 48]);

    Ok(())
}
```