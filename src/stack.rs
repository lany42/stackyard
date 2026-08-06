// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
//! Fixed-capacity stack storage.
//!
//! The [`Stack`] type keeps its elements inline, exposes its initialized
//! contents as a slice, and offers both a fast [`Stack::push`] operation that
//! silently drops values when full and [`Stack::try_push`] for recovering a
//! rejected value.
//!
//! ```rust
//! use stackyard::Stack;
//!
//! let mut stack = Stack::<&str, 2>::new();
//! stack.push("bottom");
//! stack.push("top");
//! stack.push("silently discarded");
//! assert_eq!(stack.try_push("returned"), Some("returned"));
//! assert_eq!(stack.pop(), Some("top"));
//! ```
use core::{
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    convert::{AsMut, AsRef},
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr,
    slice::SliceIndex,
};

/// A stack for holding up to `N` values of type `T`.
///
/// Stores up to `N` values inline and removes them in last-in, first-out
/// order. [`Stack::push`] silently drops a value when the stack is full;
/// [`Stack::try_push`] returns the rejected value instead.
pub struct Stack<T, const N: usize> {
    buf: [MaybeUninit<T>; N],
    top: usize,
}

impl<T, const N: usize> Stack<T, N> {
    /// Creates an empty stack with capacity `N`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            buf: [const { MaybeUninit::<T>::uninit() }; N],
            top: 0,
        }
    }

    /// Allocates an empty stack with capacity `N` in a [`Box`](alloc::boxed::Box).
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn new_boxed() -> alloc::boxed::Box<Self> {
        alloc::boxed::Box::new(Self::new())
    }

    /// Moves this stack into a [`Box`](alloc::boxed::Box).
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn into_boxed(self) -> alloc::boxed::Box<Self> {
        alloc::boxed::Box::new(self)
    }

    /// Returns the fixed number of values this stack can hold.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns the number of values currently in the stack.
    #[inline]
    pub const fn len(&self) -> usize {
        self.top
    }

    /// Returns `true` if the stack contains no values.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.top == 0
    }

    /// Returns `true` if the stack can accept another value.
    #[inline]
    pub const fn has_space(&self) -> bool {
        self.top < N
    }

    /// Returns `true` if the stack is at capacity.
    #[inline]
    pub const fn is_full(&self) -> bool {
        self.top == N
    }

    /// Attempts to push `t` onto the top of the stack.
    ///
    /// Returns `None` when `t` is stored. If the stack is full, returns
    /// `Some(t)` and leaves the stack unchanged.
    #[inline]
    pub const fn try_push(&mut self, t: T) -> Option<T> {
        if self.top >= N {
            Some(t)
        } else {
            // SAFETY:
            // - the branch above proves self.top is in bounds
            // - the slot at self.top is uninitialized and exclusively borrowed
            unsafe {
                self.buf.as_mut_ptr().add(self.top).cast::<T>().write(t);
            }
            self.top += 1;
            None
        }
    }

    /// Pushes `t` onto the top of the stack if space is available.
    ///
    /// If the stack is full, it remains unchanged and `t` is **silently
    /// dropped**. Use [`Stack::try_push`] when the rejected value must be
    /// recovered.
    ///
    /// This unit-returning fast path avoids the fallible return overhead. In
    /// this crate's benchmarks it maintains performance parity with
    /// `Vec::push` after the `Vec` has allocated sufficient capacity.
    ///
    /// # Panics
    ///
    /// Panics if the stack is full and `t`'s destructor panics while discarding
    /// it. The stack itself remains unchanged.
    #[inline]
    pub fn push(&mut self, t: T) {
        if self.top < N {
            // SAFETY:
            // - the branch above proves self.top is in bounds
            // - the slot at self.top is uninitialized and exclusively borrowed
            unsafe {
                self.buf.as_mut_ptr().add(self.top).cast::<T>().write(t);
            }
            self.top += 1;
        }
    }

    /// Removes and returns the top value, or `None` if the stack is empty.
    #[inline]
    pub const fn pop(&mut self) -> Option<T> {
        if self.top == 0 {
            None
        } else {
            self.top -= 1;

            // SAFETY:
            // - slot is initialized at self.top
            // - slot is outside valid prefix of buf
            // - slot is only read or dropped once
            unsafe { Some(self.buf[self.top].assume_init_read()) }
        }
    }

    /// Returns a reference to the top value without removing it.
    ///
    /// Returns `None` when the stack is empty.
    #[inline]
    pub const fn last(&self) -> Option<&T> {
        if self.top == 0 {
            None
        } else {
            // SAFETY:
            // - self.top is bounded by N and the branch proves self.top - 1 exists
            // - every slot below self.top is initialized
            // - MaybeUninit<T> has the same size and alignment as T
            // - the returned shared reference is tied to the borrow of self
            unsafe { Some(&*self.buf.as_ptr().add(self.top - 1).cast::<T>()) }
        }
    }

    /// Returns the initialized values in bottom-to-top order.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY:
        // - self.top starts at zero, bounded by N
        // - self.top only incremented after slots are initialized
        unsafe { self.buf[..self.top].assume_init_ref() }
    }

    /// Returns the initialized values mutably in bottom-to-top order.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: same as `as_slice`
        // - mutable borrow of self guarantees exclusive mutable borrow of slice
        unsafe { self.buf[..self.top].assume_init_mut() }
    }

    /// Clones `value` onto the stack until it reaches capacity.
    ///
    /// Existing values remain in place.
    ///
    /// # Panics
    ///
    /// Panics if [`T::clone`](Clone::clone) panics. Clones pushed before the
    /// panic remain in the stack.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        while self.has_space() {
            self.push(value.clone());
        }
    }

    // Clears the stack by dropping initialized elements and resetting the size
    // Just a straight up copy of std::Vec::clear()
    /// Removes and drops all values in the stack.
    ///
    /// # Panics
    ///
    /// Panics if a stored value's destructor panics. The stack is marked empty
    /// before values are dropped.
    #[inline]
    pub fn clear(&mut self) {
        let elems: *mut [T] = self.as_mut_slice();

        // SAFETY:
        // - `elems` comes directly from `as_mut_slice` and is therefore valid.
        // - every element in `elems` is initialized, so the entire slice may be
        //   passed to `drop_in_place`.
        // - setting `self.top` before calling `drop_in_place` means that, if an
        //   element's `Drop` impl panics, the stack's `Drop` impl does nothing
        //   (potentially leaking the rest) instead of dropping some twice.
        unsafe {
            self.top = 0;
            ptr::drop_in_place(elems);
        }
    }

    /// Copies as many values as fit from `src` onto the stack.
    ///
    /// Returns the uncopied suffix when `src` exceeds the remaining capacity.
    /// Returns `None` when `src` is empty or all of its values fit.
    pub fn copy_from_slice<'a>(&mut self, src: &'a [T]) -> Option<&'a [T]>
    where
        T: Copy,
    {
        if src.is_empty() {
            return None;
        }

        if N == self.top {
            return Some(src);
        }

        let cur = self.top;

        // Minimum of src length or remaining stack length
        let avail = src.len().min(N - cur);

        // current length + avail slots
        let new = cur + avail;
        let to_get = &src[..avail];

        // INVARIANT: both slices MUST have the same length and some data
        assert_eq!(self.buf[cur..new].len(), to_get.len());

        // Copying the source initializes every destination element.
        self.buf[cur..new].write_copy_of_slice(to_get);
        self.top = new;

        if avail == src.len() {
            None
        } else {
            Some(&src[avail..])
        }
    }

    /// Clones as many values as fit from `src` onto the stack.
    ///
    /// Returns the uncloned suffix when `src` exceeds the remaining capacity.
    /// Returns `None` when `src` is empty or all of its values fit.
    ///
    /// # Panics
    ///
    /// Panics if [`T::clone`](Clone::clone) panics. The stack retains the
    /// contents it held before this call.
    pub fn clone_from_slice<'a>(&mut self, src: &'a [T]) -> Option<&'a [T]>
    where
        T: Clone,
    {
        if src.is_empty() {
            return None;
        }

        if N == self.top {
            return Some(src);
        }

        let cur = self.top;

        // Minimum of src length or remaining stack length
        let avail = src.len().min(N - cur);

        // current length + avail slots
        let new = cur + avail;
        let to_get = &src[..avail];

        // INVARIANT: both slices MUST have the same length and some data
        assert_eq!(self.buf[cur..new].len(), to_get.len());

        // Cloning the source initializes every destination element. If a clone
        // panics, `write_clone_of_slice` drops the elements it already cloned.
        self.buf[cur..new].write_clone_of_slice(to_get);
        self.top = new;

        if avail == src.len() {
            None
        } else {
            Some(&src[avail..])
        }
    }
}

impl<T, const N: usize> Default for Stack<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone, const N: usize> Clone for Stack<T, N> {
    fn clone(&self) -> Self {
        let mut new = Self::new();
        new.clone_from_slice(self.as_slice());
        new
    }
}

impl<T, const N: usize> Drop for Stack<T, N> {
    #[inline]
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T: Hash, const N: usize> Hash for Stack<T, N> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<T: PartialOrd, const N: usize, const M: usize> PartialOrd<Stack<T, M>> for Stack<T, N> {
    #[inline]
    fn partial_cmp(&self, other: &Stack<T, M>) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Ord, const N: usize> Ord for Stack<T, N> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T, U, const N: usize, const M: usize> PartialEq<Stack<U, M>> for Stack<T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &Stack<U, M>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T, U, const N: usize> PartialEq<[U]> for Stack<T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U]) -> bool {
        self.as_slice().eq(other)
    }
}

impl<T, U, const N: usize> PartialEq<&[U]> for Stack<T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&[U]) -> bool {
        self.as_slice().eq(*other)
    }
}

impl<T, U, const N: usize, const M: usize> PartialEq<[U; M]> for Stack<T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U; M]) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<T, U, const N: usize, const M: usize> PartialEq<&[U; M]> for Stack<T, N>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&[U; M]) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<T: Eq, const N: usize> Eq for Stack<T, N> {}

impl<T, I, const N: usize> Index<I> for Stack<T, N>
where
    I: SliceIndex<[T]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self.as_slice(), index)
    }
}

impl<T, I, const N: usize> IndexMut<I> for Stack<T, N>
where
    I: SliceIndex<[T]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(self.as_mut_slice(), index)
    }
}

impl<T, const N: usize> AsRef<[T]> for Stack<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> AsMut<[T]> for Stack<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const N: usize> AsRef<Stack<T, N>> for Stack<T, N> {
    #[inline]
    fn as_ref(&self) -> &Stack<T, N> {
        self
    }
}

impl<T, const N: usize> AsMut<Stack<T, N>> for Stack<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut Stack<T, N> {
        self
    }
}

impl<T, const N: usize> Borrow<[T]> for Stack<T, N> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> BorrowMut<[T]> for Stack<T, N> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const N: usize> Deref for Stack<T, N> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const N: usize> DerefMut for Stack<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Stack<T, N> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut Stack<T, N> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::Stack;
    use std::{
        cell::Cell,
        panic::{AssertUnwindSafe, catch_unwind},
        rc::Rc,
        string::String,
        sync::atomic::{AtomicUsize, Ordering},
    };

    static COPY_CLONES: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, Eq, PartialEq)]
    struct CountedCopy(u32);

    impl Copy for CountedCopy {}

    // The observable side effect is intentional: it proves that
    // `copy_from_slice` does not accidentally regress to the cloning path.
    #[allow(clippy::non_canonical_clone_impl)]
    impl Clone for CountedCopy {
        fn clone(&self) -> Self {
            COPY_CLONES.fetch_add(1, Ordering::Relaxed);
            *self
        }
    }

    #[test]
    fn copy_from_slice_initializes_the_available_suffix_without_cloning() {
        COPY_CLONES.store(0, Ordering::Relaxed);
        let mut stack = Stack::<CountedCopy, 3>::new();
        let input = [CountedCopy(10), CountedCopy(20), CountedCopy(30)];
        assert_eq!(stack.try_push(CountedCopy(0)), None);

        assert_eq!(stack.copy_from_slice(&input), Some(&input[2..]));
        assert_eq!(
            stack.as_slice(),
            &[CountedCopy(0), CountedCopy(10), CountedCopy(20)]
        );
        assert_eq!(COPY_CLONES.load(Ordering::Relaxed), 0);

        assert_eq!(stack.copy_from_slice(&input), Some(&input[..]));
        assert_eq!(COPY_CLONES.load(Ordering::Relaxed), 0);
    }

    #[repr(align(64))]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Aligned(u8);

    #[test]
    fn slices_cover_exactly_the_initialized_aligned_prefix() {
        let mut stack = Stack::<Aligned, 3>::new();
        assert!(stack.as_slice().is_empty());
        assert_eq!(stack.try_push(Aligned(7)), None);
        assert_eq!(stack.try_push(Aligned(9)), None);

        assert_eq!(stack.as_slice(), &[Aligned(7), Aligned(9)]);
        assert_eq!(stack.last(), Some(&Aligned(9)));
        stack.as_mut_slice().swap(0, 1);
        assert_eq!(stack.as_slice(), &[Aligned(9), Aligned(7)]);
        assert_eq!(stack.last(), Some(&Aligned(7)));
    }

    struct Tracked {
        value: u8,
        drops: Rc<Cell<usize>>,
    }

    impl Tracked {
        fn new(value: u8, drops: &Rc<Cell<usize>>) -> Self {
            Self {
                value,
                drops: Rc::clone(drops),
            }
        }
    }

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[test]
    fn try_push_pop_and_drop_transfer_each_value_exactly_once() {
        let drops = Rc::new(Cell::new(0));
        let mut stack = Stack::<Tracked, 3>::new();
        for value in [10, 20, 30] {
            assert!(stack.try_push(Tracked::new(value, &drops)).is_none());
        }

        let rejected = stack
            .try_push(Tracked::new(40, &drops))
            .expect("a full stack returns the rejected value");
        assert_eq!(rejected.value, 40);
        drop(rejected);
        assert_eq!(drops.get(), 1);

        let popped = stack.pop().expect("the stack is not empty");
        assert_eq!(popped.value, 30);
        drop(popped);
        assert_eq!(drops.get(), 2);

        drop(stack);
        assert_eq!(drops.get(), 4);
    }

    #[test]
    fn push_silently_drops_a_value_rejected_by_a_full_stack() {
        let drops = Rc::new(Cell::new(0));
        let mut stack = Stack::<Tracked, 1>::new();

        stack.push(Tracked::new(10, &drops));
        assert_eq!(stack.len(), 1);
        assert_eq!(drops.get(), 0);

        stack.push(Tracked::new(20, &drops));
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.as_slice()[0].value, 10);
        assert_eq!(drops.get(), 1);

        drop(stack);
        assert_eq!(drops.get(), 2);
    }

    #[test]
    fn last_borrows_the_top_value_without_removing_it() {
        let mut stack = Stack::<u8, 2>::new();
        assert_eq!(stack.last(), None);

        stack.push(10);
        assert_eq!(stack.last(), Some(&10));
        assert_eq!(stack.len(), 1);

        stack.push(20);
        assert_eq!(stack.last(), Some(&20));
        assert_eq!(stack.len(), 2);

        assert_eq!(stack.pop(), Some(20));
        assert_eq!(stack.last(), Some(&10));
        assert_eq!(stack.pop(), Some(10));
        assert_eq!(stack.last(), None);
    }

    #[test]
    fn clone_from_slice_returns_remainder_and_stack_clone_is_independent() {
        let mut stack = Stack::<String, 3>::new();
        assert!(stack.try_push(String::from("existing")).is_none());
        let input = [
            String::from("left"),
            String::from("right"),
            String::from("remainder"),
        ];

        assert_eq!(stack.clone_from_slice(&input), Some(&input[2..]));
        assert_eq!(stack.as_slice(), ["existing", "left", "right"]);
        assert_eq!(stack.clone_from_slice(&[]), None);
        assert_eq!(stack.clone_from_slice(&input[..1]), Some(&input[..1]));

        let cloned = stack.clone();
        stack.as_mut_slice()[0].push('!');
        assert_eq!(stack.as_slice(), ["existing!", "left", "right"]);
        assert_eq!(cloned.as_slice(), ["existing", "left", "right"]);
    }

    struct PanickingClone {
        panic_on_clone: bool,
        drops: Rc<Cell<usize>>,
    }

    impl Clone for PanickingClone {
        fn clone(&self) -> Self {
            assert!(!self.panic_on_clone, "requested clone panic");
            Self {
                panic_on_clone: false,
                drops: Rc::clone(&self.drops),
            }
        }
    }

    impl Drop for PanickingClone {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[test]
    fn clone_from_slice_rolls_back_only_the_new_suffix_on_panic() {
        let drops = Rc::new(Cell::new(0));
        let mut stack = Stack::<PanickingClone, 3>::new();
        assert!(
            stack
                .try_push(PanickingClone {
                    panic_on_clone: false,
                    drops: Rc::clone(&drops),
                })
                .is_none()
        );
        let input = [
            PanickingClone {
                panic_on_clone: false,
                drops: Rc::clone(&drops),
            },
            PanickingClone {
                panic_on_clone: true,
                drops: Rc::clone(&drops),
            },
        ];

        let result = catch_unwind(AssertUnwindSafe(|| stack.clone_from_slice(&input)));

        assert!(result.is_err());
        assert_eq!(stack.len(), 1);
        assert!(!stack.as_slice()[0].panic_on_clone);
        assert_eq!(drops.get(), 1);

        drop(stack);
        assert_eq!(drops.get(), 2);
        drop(input);
        assert_eq!(drops.get(), 4);
    }

    const CONST_METHOD_RESULTS: (usize, Option<u8>, Option<u8>, bool, Option<u8>, bool, bool) = {
        let mut stack = Stack::<u8, 1>::new();
        let first_push = stack.try_push(7);
        let rejected_push = stack.try_push(9);
        let was_full = stack.is_full();
        let popped = stack.pop();
        let is_empty = stack.is_empty();
        let has_space = stack.has_space();
        let results = (
            stack.capacity(),
            first_push,
            rejected_push,
            was_full,
            popped,
            is_empty,
            has_space,
        );
        core::mem::forget(stack);
        results
    };

    const CONST_LAST_RESULTS: (Option<u8>, Option<u8>) = {
        let mut stack = Stack::<u8, 1>::new();
        let empty = match stack.last() {
            Some(value) => Some(*value),
            None => None,
        };
        let _ = stack.try_push(7);
        let populated = match stack.last() {
            Some(value) => Some(*value),
            None => None,
        };
        core::mem::forget(stack);
        (empty, populated)
    };

    #[test]
    fn const_methods_can_manipulate_stack_state() {
        assert_eq!(
            CONST_METHOD_RESULTS,
            (1, None, Some(9), true, Some(7), true, true)
        );
        assert_eq!(CONST_LAST_RESULTS, (None, Some(7)));
    }

    // Explicit RHS references exercise the corresponding `PartialEq<&...>` impls.
    #[allow(clippy::op_ref)]
    #[test]
    fn partial_eq_forwards_related_types_to_slices() {
        struct Stored(u8);
        struct Compared(u8);

        impl PartialEq<Compared> for Stored {
            fn eq(&self, other: &Compared) -> bool {
                self.0 == other.0
            }
        }

        let mut stack = Stack::<Stored, 4>::new();
        for value in [1, 2, 3] {
            assert!(stack.try_push(Stored(value)).is_none());
        }

        let array = [Compared(1), Compared(2), Compared(3)];
        let slice = array.as_slice();

        assert!(stack == *slice);
        assert!(stack == slice);
        assert!(stack == array);
        assert!(stack == &array);

        let different = [Compared(1), Compared(2), Compared(4)];
        assert!(stack != different);
        assert!(stack != &different[..2]);
    }

    #[test]
    fn zero_capacity_never_exposes_or_accepts_values() {
        let mut stack = Stack::<u8, 0>::new();

        assert_eq!(
            (
                stack.capacity(),
                stack.len(),
                stack.is_empty(),
                stack.has_space(),
                stack.is_full(),
            ),
            (0, 0, true, false, true)
        );
        assert!(stack.as_slice().is_empty());
        assert!(stack.as_mut_slice().is_empty());
        assert_eq!(stack.try_push(30), Some(30));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn fill_clones_until_capacity_and_leaves_existing_values_in_place() {
        let mut stack = Stack::<String, 3>::new();
        assert!(stack.try_push(String::from("existing")).is_none());

        stack.fill(String::from("fill"));

        assert_eq!(stack.as_slice(), ["existing", "fill", "fill"]);
    }

    struct PanicAfterOneClone {
        clone_calls: Rc<Cell<usize>>,
        drops: Rc<Cell<usize>>,
    }

    impl Clone for PanicAfterOneClone {
        fn clone(&self) -> Self {
            let clone_call = self.clone_calls.get() + 1;
            self.clone_calls.set(clone_call);
            assert_ne!(clone_call, 2, "requested clone panic");
            Self {
                clone_calls: Rc::clone(&self.clone_calls),
                drops: Rc::clone(&self.drops),
            }
        }
    }

    impl Drop for PanicAfterOneClone {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[test]
    fn fill_leaves_successful_clones_initialized_when_a_later_clone_panics() {
        let clone_calls = Rc::new(Cell::new(0));
        let drops = Rc::new(Cell::new(0));
        let mut stack = Stack::<PanicAfterOneClone, 3>::new();
        let value = PanicAfterOneClone {
            clone_calls: Rc::clone(&clone_calls),
            drops: Rc::clone(&drops),
        };

        let result = catch_unwind(AssertUnwindSafe(|| stack.fill(value)));

        assert!(result.is_err());
        assert_eq!(clone_calls.get(), 2);
        assert_eq!(stack.len(), 1);
        assert_eq!(drops.get(), 1);

        drop(stack);
        assert_eq!(drops.get(), 2);
    }

    struct PanickingDrop {
        panic_on_drop: bool,
        drops: Rc<Cell<usize>>,
    }

    impl Drop for PanickingDrop {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
            assert!(!self.panic_on_drop, "requested drop panic");
        }
    }

    #[test]
    fn clear_invalidates_the_prefix_before_a_destructor_panics() {
        let drops = Rc::new(Cell::new(0));
        let mut stack = Stack::<PanickingDrop, 3>::new();
        for panic_on_drop in [true, false, false] {
            assert!(
                stack
                    .try_push(PanickingDrop {
                        panic_on_drop,
                        drops: Rc::clone(&drops),
                    })
                    .is_none()
            );
        }

        let result = catch_unwind(AssertUnwindSafe(|| stack.clear()));

        assert!(result.is_err());
        assert!(stack.is_empty());
        let dropped_during_clear = drops.get();
        assert!(dropped_during_clear > 0);

        for _ in 0..3 {
            assert!(
                stack
                    .try_push(PanickingDrop {
                        panic_on_drop: false,
                        drops: Rc::clone(&drops),
                    })
                    .is_none()
            );
        }

        drop(stack);
        assert_eq!(drops.get(), dropped_during_clear + 3);
    }

    static ZST_DROPS: AtomicUsize = AtomicUsize::new(0);

    struct DroppedZst;

    impl Drop for DroppedZst {
        fn drop(&mut self) {
            ZST_DROPS.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn zero_sized_values_are_popped_and_cleared_exactly_once() {
        assert_eq!(core::mem::size_of::<DroppedZst>(), 0);
        ZST_DROPS.store(0, Ordering::Relaxed);
        let mut stack = Stack::<DroppedZst, 3>::new();
        for _ in 0..3 {
            assert!(stack.try_push(DroppedZst).is_none());
        }

        drop(stack.pop().expect("the stack contains three values"));
        assert_eq!(ZST_DROPS.load(Ordering::Relaxed), 1);

        stack.clear();
        assert_eq!(ZST_DROPS.load(Ordering::Relaxed), 3);
        drop(stack);
        assert_eq!(ZST_DROPS.load(Ordering::Relaxed), 3);
    }
}
