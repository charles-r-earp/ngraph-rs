use cpp::*;

use crate::Node;

cpp! {{
  #include <ngraph/op/add.hpp>
}}

cpp_class!(pub unsafe struct Add as "std::shared_ptr<ngraph::op::Add>");

impl Add {
    pub fn new<'a0, 'a1, Arg0, Arg1>(arg0: &'a0 Arg0, arg1: &'a1 Arg1) -> Self
    where
        Node: From<&'a0 Arg0> + From<&'a1 Arg1>,
    {
        let node0 = Node::from(arg0);
        let node1 = Node::from(arg1);
        cpp!(unsafe [node0 as "std::shared_ptr<ngraph::Node>", node1 as "std::shared_ptr<ngraph::Node>"] -> Add as "std::shared_ptr<ngraph::op::Add>" {
          return std::make_shared<ngraph::op::Add>(node0, node1);
        })
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
