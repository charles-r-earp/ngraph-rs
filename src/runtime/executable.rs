use super::{Tensor, TensorVector};
use cpp::*;

cpp!{{
  #include <ngraph/runtime/executable.hpp>
  #include <iostream>
}}

cpp_class!(pub unsafe struct Executable as "std::shared_ptr<ngraph::runtime::Executable>");

impl Executable {
  #[inline]
  pub(crate) fn is_null(&self) -> bool {
    cpp!(unsafe [self as "std::shared_ptr<ngraph::runtime::Executable>*"] -> bool as "bool" {
      return !bool(*self);
    })
  }
  #[inline]
  pub fn call<'o, 'i, Outputs, Inputs>(&self, outputs: Outputs, inputs: Inputs)
    where Outputs: AsRef<[&'o mut Tensor]>,
          Inputs: AsRef<[&'i Tensor]> {
    let outputs = TensorVector::from(outputs.as_ref());
    let inputs = TensorVector::from(inputs.as_ref());
    cpp!(unsafe [self as "std::shared_ptr<ngraph::runtime::Executable>*", 
                 outputs as "std::vector<std::shared_ptr<ngraph::runtime::Tensor>>", 
                 inputs as "std::vector<std::shared_ptr<ngraph::runtime::Tensor>>"] {
      //std::cout << "Output[0].get(): " << reinterpret_cast<std::size_t>(outputs[0].get()) << std::endl;
      (*self)->call(outputs, inputs);
    });
  }   
}

