// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
//! Inline-first vectors with allocator-backed spillover.
//!
//! [`SmallVec`] stores its first `N` values in an [`InlineStack`] and places
//! later values in a [`Vec`]. Values retain last-in, first-out order across the
//! two containers without requiring a heap allocation for the inline prefix.
//!
//! ```rust
//! use stackyard::SmallVec8;
//!
//! let mut values = SmallVec8::new();
//! values.push("inline");
//! values.push("also inline");
//! assert_eq!(values.pop(), Some("also inline"));
//! ```

use core::mem::size_of;

use crate::rust_alloc::vec::Vec;

use super::InlineStack;

/// A growable last-in, first-out collection with `N` inline slots.
///
/// Values fill the inline stack before spilling into an allocator-backed
/// [`Vec`]. The overflow vector is always emptied before values are removed
/// from the inline stack.
///
/// The two backing containers are disjoint, so this type deliberately does not
/// expose a single slice spanning all of its values.
///
/// Zero-sized element types are unsupported. Attempts to construct a
/// `SmallVec` with a zero-sized `T`, or to push into one whose construction was
/// bypassed, fail at compile time.
///
/// ```compile_fail
/// use stackyard::SmallVec;
///
/// let _ = SmallVec::<(), 8>::new();
/// ```
#[derive(Clone)]
pub struct SmallVec<T, const N: usize> {
    // Fields are dropped in declaration order. Keeping overflow first ensures
    // its values are destroyed before InlineStack is touched.
    overflow: Vec<T>,
    inline: InlineStack<T, N>,
}

impl<T, const N: usize> SmallVec<T, N> {
    /// Creates an empty vector with `N` inline slots and no heap reservation.
    #[inline]
    pub const fn new() -> Self {
        const {
            assert!(
                size_of::<T>() != 0,
                "SmallVec does not support zero-sized element types"
            );
        }

        Self {
            overflow: Vec::new(),
            inline: InlineStack::new(),
        }
    }

    /// Creates an empty vector with room for at least `capacity` values.
    ///
    /// Requests no larger than the inline capacity do not reserve heap storage.
    /// Larger requests reserve only the portion beyond the `N` inline slots.
    ///
    /// # Panics
    ///
    /// Panics if the requested overflow capacity exceeds [`Vec`]'s supported
    /// allocation size.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        const {
            assert!(
                size_of::<T>() != 0,
                "SmallVec does not support zero-sized element types"
            );
        }

        let overflow = if capacity > N {
            Vec::with_capacity(capacity - N)
        } else {
            Vec::new()
        };

        Self {
            overflow,
            inline: InlineStack::new(),
        }
    }

    /// Returns the number of values that can be held without another allocation.
    ///
    /// The result is the sum of the inline and overflow capacities.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inline.capacity() + self.overflow.capacity()
    }

    /// Returns the number of values in the vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.inline.len() + self.overflow.len()
    }

    /// Returns `true` if the vector contains no values.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inline.is_empty() && self.overflow.is_empty()
    }

    /// Pushes `value` onto the top of the vector.
    ///
    /// The inline stack is attempted first. Once it is full, its rejected value
    /// is forwarded to the overflow vector, which uses its normal growth policy.
    #[inline]
    pub fn push(&mut self, value: T) {
        const {
            assert!(
                size_of::<T>() != 0,
                "SmallVec does not support zero-sized element types"
            );
        }

        debug_assert!(self.overflow.is_empty() || self.inline.is_full());

        if let Some(value) = self.inline.try_push(value) {
            self.overflow.push(value);
        }

        debug_assert!(self.overflow.is_empty() || self.inline.is_full());
    }

    /// Removes and returns the top value, or `None` if the vector is empty.
    ///
    /// Overflow values are removed before the inline stack is touched.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        debug_assert!(self.overflow.is_empty() || self.inline.is_full());

        let value = match self.overflow.pop() {
            Some(value) => Some(value),
            None => self.inline.pop(),
        };

        debug_assert!(self.overflow.is_empty() || self.inline.is_full());
        value
    }

    /// Returns a reference to the top value without removing it.
    #[inline]
    pub fn last(&self) -> Option<&T> {
        debug_assert!(self.overflow.is_empty() || self.inline.is_full());

        match self.overflow.last() {
            Some(value) => Some(value),
            None => self.inline.last(),
        }
    }

    /// Returns a mutable reference to the top value without removing it.
    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        debug_assert!(self.overflow.is_empty() || self.inline.is_full());

        if self.overflow.is_empty() {
            self.inline.as_mut_slice().last_mut()
        } else {
            self.overflow.last_mut()
        }
    }

    /// Removes and drops all values while retaining allocated overflow capacity.
    ///
    /// Overflow values are cleared before the inline stack is touched.
    ///
    /// # Panics
    ///
    /// Panics if a stored value's destructor panics. If clearing an overflow
    /// value panics, the inline stack remains untouched.
    #[inline]
    pub fn clear(&mut self) {
        debug_assert!(self.overflow.is_empty() || self.inline.is_full());
        self.overflow.clear();
        self.inline.clear();
        debug_assert!(self.overflow.is_empty() || self.inline.is_full());
    }
}

impl<T, const N: usize> Default for SmallVec<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// A [`SmallVec`] with eight inline slots.
pub type SmallVec8<T> = SmallVec<T, 8>;

/// A [`SmallVec`] with sixteen inline slots.
pub type SmallVec16<T> = SmallVec<T, 16>;

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{SmallVec, SmallVec8, SmallVec16};
    use std::{cell::RefCell, rc::Rc, string::String, vec::Vec as StdVec};

    #[test]
    fn constructors_keep_both_containers_empty_and_reserve_only_overflow() {
        let new = SmallVec::<u8, 8>::new();
        assert!(new.inline.is_empty());
        assert!(new.overflow.is_empty());
        assert_eq!(new.overflow.capacity(), 0);
        assert_eq!(new.capacity(), 8);

        for requested in [0, 4, 8] {
            let values = SmallVec::<u8, 8>::with_capacity(requested);
            assert!(values.is_empty());
            assert_eq!(values.overflow.capacity(), 0);
            assert_eq!(values.capacity(), 8);
        }

        let reserved = SmallVec::<u8, 8>::with_capacity(12);
        assert!(reserved.inline.is_empty());
        assert!(reserved.overflow.is_empty());
        assert!(reserved.overflow.capacity() >= 4);
        assert!(reserved.capacity() >= 12);
    }

    #[test]
    fn pushes_and_pops_preserve_lifo_order_across_the_boundary() {
        let mut values = SmallVec::<u8, 2>::new();

        values.push(10);
        values.push(20);
        assert_eq!((values.inline.len(), values.overflow.len()), (2, 0));

        values.push(30);
        values.push(40);
        assert_eq!((values.inline.len(), values.overflow.len()), (2, 2));
        assert_eq!(values.len(), 4);
        assert_eq!(values.last(), Some(&40));

        assert_eq!(values.pop(), Some(40));
        assert_eq!((values.inline.len(), values.overflow.len()), (2, 1));
        assert_eq!(values.pop(), Some(30));
        assert_eq!((values.inline.len(), values.overflow.len()), (2, 0));
        assert_eq!(values.pop(), Some(20));
        assert_eq!(values.pop(), Some(10));
        assert_eq!(values.pop(), None);
        assert!(values.is_empty());
    }

    #[test]
    fn overflow_uses_vec_growth() {
        let mut values = SmallVec::<usize, 2>::new();

        for value in 0..32 {
            values.push(value);
        }

        assert_eq!(values.len(), 32);
        assert!(values.capacity() >= values.len());
        assert_eq!(values.inline.len(), 2);
        assert_eq!(values.overflow.len(), 30);
    }

    #[test]
    fn last_mut_follows_the_active_container() {
        let mut values = SmallVec::<u8, 2>::new();
        values.push(10);
        values.push(20);
        values.push(30);

        *values.last_mut().expect("overflow contains the top") = 31;
        assert_eq!(values.pop(), Some(31));

        *values
            .last_mut()
            .expect("inline stack now contains the top") = 21;
        assert_eq!(values.pop(), Some(21));
        assert_eq!(values.last(), Some(&10));
    }

    struct Tracked {
        id: u8,
        drops: Rc<RefCell<StdVec<u8>>>,
    }

    impl Tracked {
        fn new(id: u8, drops: &Rc<RefCell<StdVec<u8>>>) -> Self {
            Self {
                id,
                drops: Rc::clone(drops),
            }
        }
    }

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.drops.borrow_mut().push(self.id);
        }
    }

    #[test]
    fn clear_drops_overflow_first_retains_capacity_and_allows_reuse() {
        let drops = Rc::new(RefCell::new(StdVec::new()));
        let mut values = SmallVec::<Tracked, 2>::with_capacity(4);
        let capacity = values.capacity();
        for id in 1..=4 {
            values.push(Tracked::new(id, &drops));
        }

        values.clear();

        assert!(values.is_empty());
        assert_eq!(values.capacity(), capacity);
        let drops_after_clear = drops.borrow();
        assert_eq!(drops_after_clear.len(), 4);
        assert!(drops_after_clear[..2].iter().all(|id| *id > 2));
        assert!(drops_after_clear[2..].iter().all(|id| *id <= 2));
        drop(drops_after_clear);

        values.push(Tracked::new(5, &drops));
        assert_eq!((values.inline.len(), values.overflow.len()), (1, 0));
    }

    #[test]
    fn destruction_drops_overflow_before_inline() {
        let drops = Rc::new(RefCell::new(StdVec::new()));

        {
            let mut values = SmallVec::<Tracked, 2>::new();
            for id in 1..=4 {
                values.push(Tracked::new(id, &drops));
            }
        }

        let drops = drops.borrow();
        assert_eq!(drops.len(), 4);
        assert!(drops[..2].iter().all(|id| *id > 2));
        assert!(drops[2..].iter().all(|id| *id <= 2));
    }

    #[test]
    fn clone_is_independent_across_inline_and_overflow_storage() {
        let mut original = SmallVec::<String, 2>::new();
        original.push(String::from("first"));
        original.push(String::from("second"));
        original.push(String::from("third"));

        let mut cloned = original.clone();
        original
            .last_mut()
            .expect("overflow contains the top")
            .push('!');

        assert_eq!(original.pop().as_deref(), Some("third!"));
        assert_eq!(cloned.pop().as_deref(), Some("third"));
        assert_eq!(original.pop().as_deref(), Some("second"));
        assert_eq!(cloned.pop().as_deref(), Some("second"));
    }

    #[test]
    fn zero_inline_capacity_spills_every_value() {
        let mut values = SmallVec::<u8, 0>::with_capacity(2);
        assert_eq!(values.inline.capacity(), 0);

        values.push(10);
        values.push(20);

        assert_eq!((values.inline.len(), values.overflow.len()), (0, 2));
        assert_eq!(values.pop(), Some(20));
        assert_eq!(values.pop(), Some(10));
    }

    #[test]
    fn fixed_capacity_aliases_select_the_expected_inline_sizes() {
        let eight = SmallVec8::<u8>::default();
        let sixteen = SmallVec16::<u8>::default();

        assert_eq!(eight.capacity(), 8);
        assert_eq!(sixteen.capacity(), 16);
    }
}
