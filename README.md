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

- Supports GPU and CPU fallback
- Automatic gradient computation (autograd)
- Enables tensor creation with arbitrary dimension at runtime
- Ergonomic API (**note:** as the trade-off, shape checking for tensor operations happen at runtime and may panic in
  case of shape incompatibilities)

## Example

```rust
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let x = Tensor::new([1, 2], vec![1., 2.])?;
    let y = Tensor::new([1, 2], vec![3., 4.])?;
    let res = &x + &y;
    assert_eq!(res.data(), vec![4., 6.]);

    // Or use GPU (via WGPU) if you wish
    let x = Tensor::new([1, 2], vec![1., 2.])?.to_gpu()?;
    let y = Tensor::new([1, 2], vec![3., 4.])?.to_gpu()?;
    let res = &x + &y;
    assert_eq!(res.data(), vec![4., 6.]);

    Ok(())
}
```