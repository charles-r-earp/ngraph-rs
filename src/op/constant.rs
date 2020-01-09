use cpp::*;

use crate::{ElementType, Node, Shape};

cpp! {{
  #include <ngraph/op/constant.hpp>
}}

cpp_class!(pub unsafe struct Constant as "std::shared_ptr<ngraph::op::Constant>");

impl Constant {
    pub fn new<'s, 'd>(element_type: ElementType, shape: &'s Shape, data: &'d [i64]) -> Self {
        let b = data.as_ptr();
        let e = unsafe { b.add(data.len()) };
        cpp!(unsafe [element_type as "ngraph::element::Type",
                     shape as "ngraph::Shape*",
                     b as "int64_t*",
                     e as "int64_t*"
                    ] -> Constant as "std::shared_ptr<ngraph::op::Constant>" {
          std::vector<int64_t> data(b, e);
          return std::make_shared<ngraph::op::Constant>(element_type, *shape, data);
        })
    }
}

impl<'a> From<&'a Constant> for Node {
    #[inline]
    fn from(a: &'a Constant) -> Node {
        cpp!(unsafe [a as "std::shared_ptr<ngraph::op::Constant>*"] -> Node as "std::shared_ptr<ngraph::Node>" {
          return static_cast<std::shared_ptr<ngraph::Node>>(*a);
        })
    }
}
