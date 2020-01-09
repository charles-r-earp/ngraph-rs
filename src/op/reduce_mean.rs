use cpp::*;

use crate::Node;

cpp! {{
  #include <ngraph/op/reduce_mean.hpp>
}}

cpp_class!(pub unsafe struct ReduceMean as "std::shared_ptr<ngraph::op::v1::ReduceMean>");

impl ReduceMean {
    pub fn new<'a0, 'a1, Arg0, Arg1>(
        input: &'a0 Arg0,
        reduction_axes: &'a1 Arg1,
        keep_dims: bool,
    ) -> Self
    where
        Node: From<&'a0 Arg0> + From<&'a1 Arg1>,
    {
        let node0 = Node::from(input);
        let node1 = Node::from(reduction_axes);
        cpp!(unsafe [node0 as "std::shared_ptr<ngraph::Node>", node1 as "std::shared_ptr<ngraph::Node>", keep_dims as "bool"] -> ReduceMean as "std::shared_ptr<ngraph::op::v1::ReduceMean>" {
          return std::make_shared<ngraph::op::v1::ReduceMean>(node0, node1, keep_dims);
        })
    }
}

impl<'a> From<&'a ReduceMean> for Node {
    #[inline]
    fn from(a: &'a ReduceMean) -> Node {
        cpp!(unsafe [a as "std::shared_ptr<ngraph::op::v1::ReduceMean>*"] -> Node as "std::shared_ptr<ngraph::Node>" {
          return static_cast<std::shared_ptr<ngraph::Node>>(*a);
        })
    }
}
