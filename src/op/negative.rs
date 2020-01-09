use cpp::*;

use crate::Node;

cpp! {{
  #include <ngraph/op/negative.hpp>
}}

cpp_class!(pub unsafe struct Negative as "std::shared_ptr<ngraph::op::Negative>");

impl Negative {
    pub fn new<'a, Arg>(arg: &'a Arg) -> Self
    where
        Node: From<&'a Arg>,
    {
        let node = Node::from(arg);
        cpp!(unsafe [node as "std::shared_ptr<ngraph::Node>"] -> Negative as "std::shared_ptr<ngraph::op::Negative>" {
          return std::make_shared<ngraph::op::Negative>(node);
        })
    }
}

impl<'a> From<&'a Negative> for Node {
    #[inline]
    fn from(a: &'a Negative) -> Node {
        cpp!(unsafe [a as "std::shared_ptr<ngraph::op::Negative>*"] -> Node as "std::shared_ptr<ngraph::Node>" {
          return static_cast<std::shared_ptr<ngraph::Node>>(*a);
        })
    }
}
