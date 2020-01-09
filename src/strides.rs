use cpp::*;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::slice;

cpp! {{
  #include <ngraph/strides.hpp>
}}

cpp_class!(#[derive(PartialEq, Eq, PartialOrd, Ord)] pub unsafe struct Strides as "ngraph::Strides");
impl Strides {
    #[inline]
    pub fn new() -> Self {
        cpp!(unsafe [] -> Strides as "ngraph::Strides" { return ngraph::Strides(); })
    }
    #[inline]
    pub fn len(&self) -> usize {
        cpp!(unsafe [self as "const ngraph::Strides*"] -> usize as "std::size_t" {
          return self->size();
        })
    }
}

impl From<Vec<usize>> for Strides {
    #[inline]
    fn from(v: Vec<usize>) -> Self {
        let s = v.deref();
        let b = s.as_ptr();
        let e = unsafe { b.add(s.len()) };
        cpp!(unsafe [b as "std::size_t*", e as "std::size_t*"] -> Strides as "ngraph::Strides" {
          return ngraph::Strides(b, e);
        })
    }
}

#[macro_export]
macro_rules! strides {
  () => { Strides::new() };
  ($elem:expr $(, $tail:expr)*) => { Strides::from(vec![$elem, $($tail), *]) }
}

impl Deref for Strides {
    type Target = [usize];
    #[inline]
    fn deref(&self) -> &[usize] {
        let p = cpp!(unsafe [self as "const ngraph::Strides*"] -> &usize as "const std::size_t*" {
          return &self->front();
        });
        unsafe { slice::from_raw_parts(p, self.len()) }
    }
}

impl DerefMut for Strides {
    #[inline]
    fn deref_mut(&mut self) -> &mut [usize] {
        let p = cpp!(unsafe [self as "ngraph::Strides*"] -> &mut usize as "std::size_t*" {
          return &self->front();
        });
        unsafe { slice::from_raw_parts_mut(p, self.len()) }
    }
}

impl fmt::Debug for Strides {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Strides{{{}}}",
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
