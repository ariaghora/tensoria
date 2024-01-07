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

## Example

```rust
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let x = Tensor::new([1, 2], vec![1., 2.]).unwrap();
    let y = Tensor::new([1, 2], vec![3., 4.]).unwrap();
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