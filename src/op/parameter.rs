use crate::{ElementType, Shape, Node};
use std::ops::Deref;
use cpp::*;

cpp!{{
  #include <ngraph/op/parameter.hpp>
}}

cpp_class!(pub unsafe struct Parameter as "std::shared_ptr<ngraph::op::Parameter>");
/*
#[inline]
fn new_parameter<'s>(element_type: ElementType, shape: &'s Shape) -> Parameter {
  cpp!(unsafe [element_type as "ngraph::element::Type", shape as "ngraph::Shape*"] -> Parameter as "std::shared_ptr<ngraph::op::Parameter>" {
    return std::make_shared<ngraph::op::Parameter>(element_type, *shape);
  })
}*/

impl Parameter {
  #[inline]
  pub fn new<'s>(element_type: ElementType, shape: &'s Shape) -> Self {
    cpp!(unsafe [element_type as "ngraph::element::Type", shape as "ngraph::Shape*"] -> Parameter as "std::shared_ptr<ngraph::op::Parameter>" {
      return std::make_shared<ngraph::op::Parameter>(element_type, *shape);
    })
  }
}

impl<'a> From<&'a Parameter> for Node {
  #[inline]
  fn from(p: &'a Parameter) -> Node {
    cpp!(unsafe [p as "std::shared_ptr<ngraph::op::Parameter>*"] -> Node as "std::shared_ptr<ngraph::Node>" {
      return static_cast<std::shared_ptr<ngraph::Node>>(*p);
    })
  }
}

cpp_class!(pub unsafe struct ParameterVector as "ngraph::ParameterVector");

impl<'a, 'p> From<&'a [&'p Parameter]> for ParameterVector { 
  #[inline]
  fn from(parameters: &'a [&'p Parameter]) -> Self {
    let parameters: Vec<Parameter> = parameters.iter()
                                               .map(|&p| p.clone())
                                               .collect();
    let b = parameters.deref().as_ptr();
    let e = unsafe { b.add(parameters.len()) };
    cpp!(unsafe [b as "std::shared_ptr<ngraph::op::Parameter>*", e as "std::shared_ptr<ngraph::op::Parameter>*"] -> ParameterVector as "ngraph::ParameterVector" {
      return ngraph::ParameterVector(b, e);
    })
  }
}
