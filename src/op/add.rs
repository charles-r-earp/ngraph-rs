use crate::Node;
use cpp::*;

cpp! {{
  #include <ngraph/op/add.hpp>
}}

cpp_class!(pub unsafe struct Add as "std::shared_ptr<ngraph::op::Add>");

fn new_add<'a0, 'a1, Arg0, Arg1>(arg0: &'a0 Arg0, arg1: &'a1 Arg1) -> Add
where
    Node: From<&'a0 Arg0> + From<&'a1 Arg1>,
{
    let arg0 = Node::from(arg0);
    let arg1 = Node::from(arg1);
    cpp!(unsafe [arg0 as "std::shared_ptr<ngraph::Node>", arg1 as "std::shared_ptr<ngraph::Node>"] -> Add as "std::shared_ptr<ngraph::op::Add>" {
      return std::make_shared<ngraph::op::Add>(arg0, arg1);
    })
}

impl Add {
    pub fn new<'a0, 'a1, Arg0, Arg1>(arg0: &'a0 Arg0, arg1: &'a1 Arg1) -> Self
    where
        Node: From<&'a0 Arg0> + From<&'a1 Arg1>,
    {
        new_add(arg0, arg1)
    }
}

impl<'a> From<&'a Add> for Node {
    #[inline]
    fn from(a: &'a Add) -> Node {
        cpp!(unsafe [a as "std::shared_ptr<ngraph::op::Add>*"] -> Node as "std::shared_ptr<ngraph::Node>" {
          return static_cast<std::shared_ptr<ngraph::Node>>(*a);
        })
    }
}
