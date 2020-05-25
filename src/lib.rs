#![recursion_limit = "1024"]

mod element_type;
mod function;
mod node;
pub mod op;
pub mod runtime;
mod shape_inference;
mod shapes;
mod strides;

pub use self::element_type::ElementType;
pub use self::function::{create_function, Function};
pub use self::node::{Node, NodeVector};
pub use self::op::{Parameter, ParameterVector, RoundingType};
pub use self::shape_inference::*;
pub use self::shapes::Shape;
pub use self::strides::Strides;

pub mod prelude {
    pub use crate::{
        create_function, runtime::Backend, shape, shape_inference, ElementType, Shape,
    };
}
