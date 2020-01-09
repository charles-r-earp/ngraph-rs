use cpp::*;

use crate::{op::RoundingType, Node, Shape, Strides};

cpp! {{
  #include <ngraph/op/max_pool.hpp>
}}

cpp_class!(pub unsafe struct MaxPool as "std::shared_ptr<ngraph::op::v1::MaxPool>");

impl MaxPool {
    pub fn new<'a, Arg>(
        node_ref: &'a Arg,
        strides: &Strides,
        pads_begin: &Shape,
        pads_end: &Shape,
        kernel: &Shape,
        rounding_mode: RoundingType,
    ) -> Self
    where
        Node: From<&'a Arg>,
    {
        let node = Node::from(node_ref);
        cpp!(unsafe [node as "std::shared_ptr<ngraph::Node>",
                     strides as "ngraph::Strides*",
                     pads_begin as "ngraph::Shape*",
                     pads_end as "ngraph::Shape*",
                     kernel as "ngraph::Shape*",
                     rounding_mode as "ngraph::op::RoundingType"] -> MaxPool as "std::shared_ptr<ngraph::op::v1::MaxPool>" {
          return std::make_shared<ngraph::op::v1::MaxPool>(
            node, *strides, *pads_begin, *pads_end, *kernel, rounding_mode
          );
        })
    }
}

impl<'a> From<&'a MaxPool> for Node {
    #[inline]
    fn from(a: &'a MaxPool) -> Node {
        cpp!(unsafe [a as "std::shared_ptr<ngraph::op::v1::MaxPool>*"] -> Node as "std::shared_ptr<ngraph::Node>" {
          return static_cast<std::shared_ptr<ngraph::Node>>(*a);
        })
    }
}
