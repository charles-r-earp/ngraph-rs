use crate::Strides;
use cpp::*;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::slice;

cpp! {{
  #include <ngraph/shape.hpp>
  #include <sstream>
}}

cpp_class!(#[derive(PartialEq, Eq, PartialOrd, Ord)] pub unsafe struct Shape as "ngraph::Shape");
impl Shape {
    #[inline]
    pub fn new() -> Self {
        cpp!(unsafe [] -> Shape as "ngraph::Shape" { return ngraph::Shape(); })
    }
    #[inline]
    pub fn len(&self) -> usize {
        cpp!(unsafe [self as "const ngraph::Shape*"] -> usize as "std::size_t" {
          return self->size();
        })
    }
    #[inline]
    pub fn shape_size(&self) -> usize {
        cpp!(unsafe [self as "ngraph::Shape*"] -> usize as "std::size_t" {
          return shape_size(*self);
        })
    }
    #[inline]
    pub fn row_major_strides(&self) -> Strides {
        cpp!(unsafe [self as "ngraph::Shape*"] -> Strides as "ngraph::Strides" {
          return row_major_strides(*self);
        })
    }
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn is_vector(&self) -> bool {
        self.len() == 1
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        let s = v.deref();
        let b = s.as_ptr();
        let e = unsafe { b.add(s.len()) };
        cpp!(unsafe [b as "std::size_t*", e as "std::size_t*"] -> Shape as "ngraph::Shape" {
          return ngraph::Shape(b, e);
        })
    }
}

#[macro_export]
macro_rules! shape {
  () => { Shape::new() };
  ($elem:expr $(, $tail:expr)*) => { Shape::from(vec![$elem, $($tail), *]) }
}

impl Deref for Shape {
    type Target = [usize];
    #[inline]
    fn deref(&self) -> &[usize] {
        let p = cpp!(unsafe [self as "const ngraph::Shape*"] -> &usize as "const std::size_t*" {
          return &self->front();
        });
        unsafe { slice::from_raw_parts(p, self.len()) }
    }
}

impl DerefMut for Shape {
    #[inline]
    fn deref_mut(&mut self) -> &mut [usize] {
        let p = cpp!(unsafe [self as "ngraph::Shape*"] -> &mut usize as "std::size_t*" {
          return &self->front();
        });
        unsafe { slice::from_raw_parts_mut(p, self.len()) }
    }
}

impl fmt::Debug for Shape {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Shape{{{}}}",
            self.iter().fold(String::new(), |acc, x| {
                if acc.is_empty() {
                    format!("{:?}", x)
                } else {
                    acc + &format!(", {:?}", x)
                }
            })
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{shape, strides, Shape, Strides};
    #[test]
    fn test_shape_size() {
        assert_eq!(1, shape![].shape_size());
        assert_eq!(2 * 3 * 5, shape![2, 3, 5].shape_size());
    }
    #[test]
    fn test_row_major_strides() {
        assert_eq!(strides![], shape![].row_major_strides());
        assert_eq!(strides![1], shape![3].row_major_strides());
        assert_eq!(strides![7, 1], shape![2, 7].row_major_strides());
        assert_eq!(strides![84, 12, 1], shape![5, 7, 12].row_major_strides());
    }
}
