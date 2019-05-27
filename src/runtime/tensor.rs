use std::mem;
use cpp::*;

cpp!{{
  #include <ngraph/runtime/tensor.hpp>
  #include <iostream>
}}

cpp_class!(pub unsafe struct Tensor as "std::shared_ptr<ngraph::runtime::Tensor>");

impl Tensor {
  #[inline]
  pub fn ptr(&self) -> usize {
    cpp!(unsafe [self as "std::shared_ptr<ngraph::runtime::Tensor>*"] -> usize as "std::size_t" {
      return reinterpret_cast<std::size_t>(self->get());
    })
  } 
  #[inline]
  pub fn write<'d, T>(&self, offset: usize, data: &'d [T]) {
    let p = data.as_ptr();
    let n = data.len() * mem::size_of::<T>();
    cpp!(unsafe [self as "std::shared_ptr<ngraph::runtime::Tensor>*", p as "const void*", offset as "std::size_t", n as "std::size_t"] {
      (*self)->write(p, offset, n);
    });
  }
  #[inline]
  pub fn read<'d, T>(&self, offset: usize, data: &'d mut [T]) {
    let p = data.as_mut_ptr();
    let n = data.len() * mem::size_of::<T>();
    cpp!(unsafe [self as "std::shared_ptr<ngraph::runtime::Tensor>*", p as "void*", offset as "std::size_t", n as "std::size_t"] {
      (*self)->read(p, offset, n);
    });
  }
}

cpp_class!(pub unsafe struct TensorVector as "std::vector<std::shared_ptr<ngraph::runtime::Tensor>>");

impl<'a> From<&'a [Tensor]> for TensorVector {
  #[inline]
  fn from(tensors: &'a [Tensor]) -> Self {
    let b = tensors.as_ptr();
    let e = unsafe { b.add(tensors.len()) };
    cpp!(unsafe [b as "std::shared_ptr<ngraph::runtime::Tensor>*", e as "std::shared_ptr<ngraph::runtime::Tensor>*"] -> TensorVector as "std::vector<std::shared_ptr<ngraph::runtime::Tensor>>" {
      return std::vector<std::shared_ptr<ngraph::runtime::Tensor>>(b, e);
    })
  }
}

impl<'a, 't> From<&'a [&'t Tensor]> for TensorVector {
  #[inline]
  fn from(tensors: &'a [&'t Tensor]) -> Self {
    let tensors: Vec<Tensor> = tensors.iter()
                                     .map(|&t| t.clone())
                                     .collect();
    tensors.as_slice().into()
  }
}

impl<'a, 't> From<&'a [&'t mut Tensor]> for TensorVector {
  #[inline]
  fn from(tensors: &'a [&'t mut Tensor]) -> Self {
    let tensors: Vec<Tensor> = tensors.iter()
                                     .map(|t| (*t).clone())
                                     .collect();
    tensors.as_slice().into()
  }
}

