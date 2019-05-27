use crate::{Node};
use cpp::*;

cpp!{{
  #include <ngraph/op/multiply.hpp>
}}

cpp_class!(pub unsafe struct Multiply as "std::shared_ptr<ngraph::op::Multiply>");

fn new_multiply<'a0,'a1, Arg0, Arg1>(arg0: &'a0 Arg0, arg1: &'a1 Arg1) -> Multiply
  where Node: From<&'a0 Arg0> + From<&'a1 Arg1> {
  let arg0 = Node::from(arg0);
  let arg1 = Node::from(arg1);
  cpp!(unsafe [arg0 as "std::shared_ptr<ngraph::Node>", arg1 as "std::shared_ptr<ngraph::Node>"] -> Multiply as "std::shared_ptr<ngraph::op::Multiply>" {
    return std::make_shared<ngraph::op::Multiply>(arg0, arg1);
  })
}

impl Multiply {
  pub fn new<'a0,'a1, Arg0, Arg1>(arg0: &'a0 Arg0, arg1: &'a1 Arg1) -> Self
    where Node: From<&'a0 Arg0> + From<&'a1 Arg1> {
    new_multiply(arg0, arg1)
  }
}

impl<'a> From<&'a Multiply> for Node {
  #[inline]
  fn from(a: &'a Multiply) -> Node {
    cpp!(unsafe [a as "std::shared_ptr<ngraph::op::Multiply>*"] -> Node as "std::shared_ptr<ngraph::Node>" {
      return static_cast<std::shared_ptr<ngraph::Node>>(*a);
    })
  }
}
