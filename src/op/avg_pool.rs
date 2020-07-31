use cpp::*;

use crate::{op::RoundingType, Node, Shape, Strides};

cpp! {{
  #include <ngraph/op/avg_pool.hpp>
}}

cpp_class!(pub unsafe struct AvgPool as "std::shared_ptr<ngraph::op::v1::AvgPool>");

impl AvgPool {
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
                     rounding_mode as "ngraph::op::RoundingType"] -> AvgPool as "std::shared_ptr<ngraph::op::v1::AvgPool>" {
          return std::make_shared<ngraph::op::v1::AvgPool>(
            node, *strides, *pads_begin, *pads_end, *kernel, true, rounding_mode
          );
        })
    }
}

impl<'a> From<&'a AvgPool> for Node {
    #[inline]
    fn from(a: &'a AvgPool) -> Node {
        cpp!(unsafe [a as "std::shared_ptr<ngraph::op::v1::AvgPool>*"] -> Node as "std::shared_ptr<ngraph::Node>" {
          return static_cast<std::shared_ptr<ngraph::Node>>(*a);
        })
    }
}
