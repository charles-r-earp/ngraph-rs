use cpp::*;

cpp! {{
  #include <ngraph/node.hpp>
}}

cpp_class!(pub unsafe struct Node as "std::shared_ptr<ngraph::Node>");

cpp_class!(pub unsafe struct NodeVector as "ngraph::NodeVector");

impl<'a, 'n> From<&'a [&'n Node]> for NodeVector {
    #[inline]
    fn from(nodes: &'a [&'n Node]) -> Self {
        let nodes: Vec<Node> = nodes.iter().map(|&p| p.clone()).collect();
        let b = nodes.as_ptr();
        let e = unsafe { b.add(nodes.len()) };
        cpp!(unsafe [b as "std::shared_ptr<ngraph::Node>*", e as "std::shared_ptr<ngraph::Node>*"] -> NodeVector as "ngraph::NodeVector" {
          return ngraph::NodeVector(b, e);
        })
    }
}
