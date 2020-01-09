use cpp::*;

use crate::{Node, NodeVector};

cpp! {{
  #include <ngraph/op/concat.hpp>
}}

cpp_class!(pub unsafe struct Concat as "std::shared_ptr<ngraph::op::Concat>");

impl Concat {
    pub fn new<'n, Nodes>(nodes: Nodes, axis: i64) -> Self
    where
        Nodes: AsRef<[&'n Node]>,
    {
        let nodes = NodeVector::from(nodes.as_ref());
        cpp!(unsafe [nodes as "ngraph::NodeVector", axis as "uint64_t"] -> Concat as "std::shared_ptr<ngraph::op::Concat>" {
          return std::make_shared<ngraph::op::Concat>(nodes, axis);
        })
    }
}

impl<'n> From<&'n Concat> for Node {
    #[inline]
    fn from(a: &'n Concat) -> Node {
        cpp!(unsafe [a as "std::shared_ptr<ngraph::op::Concat>*"] -> Node as "std::shared_ptr<ngraph::Node>" {
          return static_cast<std::shared_ptr<ngraph::Node>>(*a);
        })
    }
}
