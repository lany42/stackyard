// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
//! Fixed-capacity last-in, first-out collections.
//!
//! [`InlineStack`] owns an inline array with a compile-time capacity. [`Stack`]
//! holds a small handle to a run-time-sized lease borrowed from an [`Alloc`].
//! Both expose their initialized values as a bottom-to-top slice and never grow.
//! [`InlineStack::push`] and [`Stack::push`] silently discard values when full;
//! [`InlineStack::try_push`] and [`Stack::try_push`] return rejected values.
//!
//! A [`TypedBlock`](crate::TypedBlock) is the simplest backing allocator when
//! the element type and maximum capacity are known.
//!
//! ```rust
//! use stackyard::{Stack, TypedBlock};
//!
//! let block = TypedBlock::<u8, 4>::new();
//! let mut stack = Stack::<u8, _>::try_new_in(3, &block).unwrap();
//! stack.push(1);
//! stack.push(2);
//! assert_eq!(stack.pop(), Some(2));
//! ```

use core::{
    alloc::Layout,
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    convert::{AsMut, AsRef},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr::{self, NonNull},
    slice::{self, SliceIndex},
};

use crate::Alloc;

mod inline_stack;
#[cfg(feature = "alloc")]
mod small_vec;
mod vec;

pub use inline_stack::InlineStack;
#[cfg(feature = "alloc")]
pub use small_vec::{SmallVec, SmallVec8, SmallVec16};
pub use vec::Vector;

/// A fixed-capacity stack backed by a borrowed allocator.
///
/// The stack obtains one positive-sized allocation during construction and
/// retains it until destruction. Its capacity never changes, and steady-state
/// operations do not call the allocator. Its initialized values form a
/// bottom-to-top prefix that can be accessed through slice operations.
///
/// `Stack::push` silently drops a value when the stack is full, matching
/// [`InlineStack::push`]; [`Stack::try_push`] returns the rejected value
/// instead. Dropping the stack destroys its initialized values and releases
/// its allocation.
///
/// Zero capacity and zero-sized element types are unsupported because
/// [`Alloc`] rejects zero-sized layouts.
pub struct Stack<'a, T, A: Alloc + ?Sized> {
    allocator: &'a A,
    ptr: NonNull<MaybeUninit<T>>,
    top: usize,
    capacity: usize,
    _owns: PhantomData<T>,
}

impl<'a, T, A: Alloc + ?Sized> Stack<'a, T, A> {
    /// Attempts to allocate an empty stack with `capacity` slots.
    ///
    /// The stack borrows `allocator` until it is dropped. Returns `None` if the
    /// array layout overflows or the allocator rejects the request, including
    /// when its storage is already leased or too small. Because allocators
    /// reject zero-sized layouts, zero capacity and zero-sized element types
    /// are unsupported.
    #[inline]
    pub fn try_new_in(capacity: usize, allocator: &'a A) -> Option<Self> {
        let layout = Layout::array::<T>(capacity).ok()?;
        let ptr = allocator.alloc(layout)?.cast::<MaybeUninit<T>>();

        Some(Self {
            allocator,
            ptr,
            top: 0,
            capacity,
            _owns: PhantomData,
        })
    }

    /// Moves this stack handle into a [`Box`](crate::rust_alloc::boxed::Box).
    ///
    /// The backing allocation remains owned by `allocator`; boxing moves only
    /// the handle.
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn into_boxed(self) -> crate::rust_alloc::boxed::Box<Self> {
        crate::rust_alloc::boxed::Box::new(self)
    }

    /// Moves the stack's values into a freshly allocated
    /// [`Vec`](crate::rust_alloc::vec::Vec).
    ///
    /// The vector is allocated with capacity for every stack slot and
    /// preserves the values' bottom-to-top order. The stack's borrowed raw
    /// allocation is released before this function returns.
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn into_vec(mut self) -> crate::rust_alloc::vec::Vec<T> {
        let len = self.top;
        let mut vec = crate::rust_alloc::vec::Vec::with_capacity(self.capacity);

        // SAFETY:
        // - `len <= capacity`, and every source slot below `len` is initialized
        // - `vec` has capacity for at least `capacity` values
        // - the source lease and the vector allocation do not overlap
        // - the copy initializes the vector prefix before `set_len`
        unsafe {
            ptr::copy_nonoverlapping(self.ptr.as_ptr().cast::<T>(), vec.as_mut_ptr(), len);
            vec.set_len(len);
        }

        // Ownership of the initialized values now belongs to `vec`. Dropping
        // the emptied Stack releases only its raw allocation.
        self.top = 0;
        vec
    }

    /// Returns the fixed number of values this stack can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of values currently in the stack.
    #[inline]
    pub fn len(&self) -> usize {
        self.top
    }

    /// Returns `true` if the stack contains no values.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.top == 0
    }

    /// Returns `true` if the stack can accept another value.
    #[inline]
    pub fn has_space(&self) -> bool {
        self.top < self.capacity
    }

    /// Returns `true` if the stack is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.top == self.capacity
    }

    /// Attempts to push `value` onto the top of the stack.
    ///
    /// Returns `None` when `value` is stored. If the stack is full, returns
    /// `Some(value)` and leaves the stack unchanged.
    #[inline]
    pub fn try_push(&mut self, value: T) -> Option<T> {
        let top = self.top;

        if top >= self.capacity {
            Some(value)
        } else {
            // SAFETY:
            // - construction obtained storage for exactly `capacity` values
            // - the branch proves `top` is within that allocation
            // - slots below `top` are initialized and this slot is not
            // - `&mut self` grants exclusive access through the allocation
            unsafe {
                let end = self.ptr.as_ptr().add(top).cast::<T>();
                ptr::write(end, value);
                self.top = top + 1;
            }
            None
        }
    }

    /// Pushes `value` onto the stack if space is available.
    ///
    /// If the stack is full, it remains unchanged and `value` is silently
    /// dropped.
    ///
    /// # Panics
    ///
    /// Panics if the stack is full and `value`'s destructor panics while
    /// discarding it. The stack itself remains unchanged.
    #[inline]
    pub fn push(&mut self, value: T) {
        let top = self.top;

        if top < self.capacity {
            // SAFETY:
            // - construction obtained storage for exactly `capacity` values
            // - the branch proves `top` is within that allocation
            // - slots below `top` are initialized and this slot is not
            // - `&mut self` grants exclusive access through the allocation
            unsafe {
                let end = self.ptr.as_ptr().add(top).cast::<T>();
                ptr::write(end, value);
                self.top = top + 1;
            }
        }
    }

    /// Removes and returns the top value, or `None` if the stack is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        let top = self.top;

        if top == 0 {
            None
        } else {
            let new_top = top - 1;
            self.top = new_top;

            // SAFETY:
            // - the slot at `new_top` was initialized
            // - decrementing `top` removes it from the initialized prefix
            // - the value is read exactly once and remains owned by the caller
            unsafe { Some(self.ptr.as_ptr().add(new_top).cast::<T>().read()) }
        }
    }

    /// Returns a reference to the top value without removing it.
    ///
    /// Returns `None` when the stack is empty.
    #[inline]
    pub fn last(&self) -> Option<&T> {
        if self.top == 0 {
            None
        } else {
            // SAFETY:
            // - `top <= capacity`, and the branch proves `top - 1` exists
            // - every slot below `top` is initialized
            // - the returned reference is tied to the shared borrow of `self`
            unsafe { Some(&*self.ptr.as_ptr().add(self.top - 1).cast::<T>()) }
        }
    }

    /// Returns the initialized values in bottom-to-top order.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY:
        // - construction obtained storage for `capacity` aligned values
        // - `top` never exceeds `capacity`
        // - exactly the first `top` slots are initialized
        // - the returned reference is tied to the shared borrow of `self`
        unsafe { slice::from_raw_parts(self.ptr.as_ptr().cast::<T>(), self.top) }
    }

    /// Returns the initialized values mutably in bottom-to-top order.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY:
        // - construction obtained storage for `capacity` aligned values
        // - `top` never exceeds `capacity`
        // - exactly the first `top` slots are initialized
        // - the returned reference is tied to the exclusive borrow of `self`
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr().cast::<T>(), self.top) }
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

    /// Removes and drops all values in the stack.
    ///
    /// # Panics
    ///
    /// Panics if a stored value's destructor panics. The stack is marked empty
    /// before values are dropped.
    #[inline]
    pub fn clear(&mut self) {
        let elements: *mut [T] = self.as_mut_slice();

        // SAFETY:
        // - `elements` is exactly the initialized prefix
        // - resetting `top` first transfers responsibility for that prefix to
        //   this drop operation and prevents any later Stack drop from retrying
        // - compiler-generated slice drop glue handles element destruction
        unsafe {
            self.top = 0;
            ptr::drop_in_place(elements);
        }
    }

    /// Copies as many values as fit from `src` onto the stack.
    ///
    /// Returns the uncopied suffix when `src` exceeds the remaining capacity.
    /// Returns `None` when `src` is empty or all of its values fit.
    pub fn copy_from_slice<'b>(&mut self, src: &'b [T]) -> Option<&'b [T]>
    where
        T: Copy,
    {
        if src.is_empty() {
            return None;
        }

        if self.is_full() {
            return Some(src);
        }

        let current = self.top;
        let available = src.len().min(self.capacity - current);
        let new_top = current + available;
        let source = &src[..available];

        // SAFETY:
        // - `available <= capacity - current`, so this range is in the lease
        // - the range starts at `top`, so every destination slot is uninitialized
        // - `&mut self` grants exclusive access to the destination range
        let destination =
            unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr().add(current), available) };
        destination.write_copy_of_slice(source);
        self.top = new_top;

        if available == src.len() {
            None
        } else {
            Some(&src[available..])
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
    pub fn clone_from_slice<'b>(&mut self, src: &'b [T]) -> Option<&'b [T]>
    where
        T: Clone,
    {
        if src.is_empty() {
            return None;
        }

        if self.is_full() {
            return Some(src);
        }

        let current = self.top;
        let available = src.len().min(self.capacity - current);
        let new_top = current + available;
        let source = &src[..available];

        // SAFETY:
        // - `available <= capacity - current`, so this range is in the lease
        // - the range starts at `top`, so every destination slot is uninitialized
        // - `&mut self` grants exclusive access to the destination range
        let destination =
            unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr().add(current), available) };

        // If cloning panics, this operation drops its partially cloned prefix
        // and leaves `top` unchanged.
        destination.write_clone_of_slice(source);
        self.top = new_top;

        if available == src.len() {
            None
        } else {
            Some(&src[available..])
        }
    }

    /// Clones this stack into a lease from `allocator`.
    ///
    /// The destination preserves this stack's capacity and initialized prefix.
    /// Returns `None` without cloning any values if the destination layout
    /// overflows or `allocator` cannot grant a disjoint lease.
    ///
    /// # Panics
    ///
    /// Panics if [`T::clone`](Clone::clone) panics. The partially constructed
    /// destination drops every successful clone and releases its lease while
    /// unwinding; this source stack remains unchanged.
    pub fn try_clone_into<'b, B>(&self, allocator: &'b B) -> Option<Stack<'b, T, B>>
    where
        T: Clone,
        B: Alloc + ?Sized,
    {
        let mut cloned = Stack::try_new_in(self.capacity, allocator)?;
        let remainder = cloned.clone_from_slice(self.as_slice());
        debug_assert!(remainder.is_none());
        Some(cloned)
    }
}

impl<'a, T: Hash, A: Alloc + ?Sized> Hash for Stack<'a, T, A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<'a, 'b, T: PartialOrd, A: Alloc + ?Sized, B: Alloc + ?Sized> PartialOrd<Stack<'b, T, B>>
    for Stack<'a, T, A>
{
    #[inline]
    fn partial_cmp(&self, other: &Stack<'b, T, B>) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<'a, T: Ord, A: Alloc + ?Sized> Ord for Stack<'a, T, A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<'a, 'b, T, U, A: Alloc + ?Sized, B: Alloc + ?Sized> PartialEq<Stack<'b, U, B>>
    for Stack<'a, T, A>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &Stack<'b, U, B>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a, T, U, A: Alloc + ?Sized> PartialEq<[U]> for Stack<'a, T, A>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U]) -> bool {
        self.as_slice().eq(other)
    }
}

impl<'a, T, U, A: Alloc + ?Sized> PartialEq<&[U]> for Stack<'a, T, A>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&[U]) -> bool {
        self.as_slice().eq(*other)
    }
}

impl<'a, T, U, A: Alloc + ?Sized, const N: usize> PartialEq<[U; N]> for Stack<'a, T, A>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<'a, T, U, A: Alloc + ?Sized, const N: usize> PartialEq<&[U; N]> for Stack<'a, T, A>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&[U; N]) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<'a, T: Eq, A: Alloc + ?Sized> Eq for Stack<'a, T, A> {}

impl<'a, T, A: Alloc + ?Sized, I> Index<I> for Stack<'a, T, A>
where
    I: SliceIndex<[T]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self.as_slice(), index)
    }
}

impl<'a, T, A: Alloc + ?Sized, I> IndexMut<I> for Stack<'a, T, A>
where
    I: SliceIndex<[T]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(self.as_mut_slice(), index)
    }
}

impl<'a, T, A: Alloc + ?Sized> AsRef<[T]> for Stack<'a, T, A> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T, A: Alloc + ?Sized> AsMut<[T]> for Stack<'a, T, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T, A: Alloc + ?Sized> AsRef<Stack<'a, T, A>> for Stack<'a, T, A> {
    #[inline]
    fn as_ref(&self) -> &Stack<'a, T, A> {
        self
    }
}

impl<'a, T, A: Alloc + ?Sized> AsMut<Stack<'a, T, A>> for Stack<'a, T, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut Stack<'a, T, A> {
        self
    }
}

impl<'a, T, A: Alloc + ?Sized> Borrow<[T]> for Stack<'a, T, A> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T, A: Alloc + ?Sized> BorrowMut<[T]> for Stack<'a, T, A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T, A: Alloc + ?Sized> Deref for Stack<'a, T, A> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, T, A: Alloc + ?Sized> DerefMut for Stack<'a, T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'stack, 'alloc, T, A: Alloc + ?Sized> IntoIterator for &'stack Stack<'alloc, T, A> {
    type Item = &'stack T;
    type IntoIter = core::slice::Iter<'stack, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'stack, 'alloc, T, A: Alloc + ?Sized> IntoIterator for &'stack mut Stack<'alloc, T, A> {
    type Item = &'stack mut T;
    type IntoIter = core::slice::IterMut<'stack, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

struct ReleaseGuard<'a, A: Alloc + ?Sized> {
    allocator: &'a A,
    ptr: NonNull<u8>,
    layout: Layout,
}

impl<A: Alloc + ?Sized> Drop for ReleaseGuard<'_, A> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: the guard is created from the Stack's one live lease with
        // its exact layout and is the sole owner of the release obligation.
        unsafe { self.allocator.free(self.ptr, self.layout) };
    }
}

impl<T, A: Alloc + ?Sized> Drop for Stack<'_, T, A> {
    #[inline]
    fn drop(&mut self) {
        let layout = Layout::array::<T>(self.capacity)
            .expect("Stack capacity produced a valid layout during construction");

        let release = ReleaseGuard {
            allocator: self.allocator,
            ptr: self.ptr.cast::<u8>(),
            layout,
        };

        // `release` keeps the allocation live while `clear` drops values and
        // also releases it if an element destructor unwinds through this frame.
        self.clear();

        // This is the only release path. If element destruction unwinds, the
        // same call occurs automatically while unwinding instead.
        drop(release);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::TypedBlock;
    use core::{
        borrow::{Borrow, BorrowMut},
        cell::Cell,
        hash::{Hash, Hasher},
    };
    use std::{
        collections::hash_map::DefaultHasher,
        panic::{AssertUnwindSafe, catch_unwind},
        rc::Rc,
        string::String,
        sync::atomic::{AtomicUsize, Ordering},
        vec::Vec,
    };

    use super::Stack;

    static COPY_CLONES: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, Eq, PartialEq)]
    struct CountedCopy(u32);

    impl Copy for CountedCopy {}

    #[allow(clippy::non_canonical_clone_impl)]
    impl Clone for CountedCopy {
        fn clone(&self) -> Self {
            COPY_CLONES.fetch_add(1, Ordering::Relaxed);
            *self
        }
    }

    struct SmokeTracked {
        value: u8,
        drops: Rc<Cell<usize>>,
    }

    impl SmokeTracked {
        fn new(value: u8, drops: &Rc<Cell<usize>>) -> Self {
            Self {
                value,
                drops: Rc::clone(drops),
            }
        }
    }

    impl Drop for SmokeTracked {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[test]
    fn copy_from_slice_initializes_the_available_suffix_without_cloning() {
        COPY_CLONES.store(0, Ordering::Relaxed);
        let block = TypedBlock::<CountedCopy, 3>::new();
        let mut stack = Stack::try_new_in(3, &block).expect("three values fit");
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

    #[test]
    fn try_push_pop_and_drop_transfer_each_value_exactly_once() {
        let drops = Rc::new(Cell::new(0));
        let block = TypedBlock::<SmokeTracked, 3>::new();
        let mut stack = Stack::try_new_in(3, &block).expect("three values fit");

        assert_eq!(
            (
                stack.capacity(),
                stack.len(),
                stack.is_empty(),
                stack.has_space(),
                stack.is_full(),
            ),
            (3, 0, true, true, false)
        );
        assert!(stack.as_slice().is_empty());
        assert!(stack.last().is_none());

        assert!(stack.try_push(SmokeTracked::new(10, &drops)).is_none());
        stack.push(SmokeTracked::new(20, &drops));
        assert!(stack.try_push(SmokeTracked::new(30, &drops)).is_none());
        assert_eq!(
            stack
                .as_slice()
                .iter()
                .map(|tracked| tracked.value)
                .collect::<Vec<_>>(),
            [10, 20, 30]
        );
        assert_eq!(stack.last().map(|tracked| tracked.value), Some(30));
        assert_eq!(
            (
                stack.len(),
                stack.is_empty(),
                stack.has_space(),
                stack.is_full(),
            ),
            (3, false, false, true)
        );

        let rejected = stack
            .try_push(SmokeTracked::new(40, &drops))
            .expect("a full stack returns the rejected value");
        assert_eq!(rejected.value, 40);
        drop(rejected);
        assert_eq!(drops.get(), 1);

        stack.push(SmokeTracked::new(50, &drops));
        assert_eq!(drops.get(), 2);
        assert_eq!(
            stack
                .as_slice()
                .iter()
                .map(|tracked| tracked.value)
                .collect::<Vec<_>>(),
            [10, 20, 30]
        );

        let popped = stack.pop().expect("the stack is not empty");
        assert_eq!(popped.value, 30);
        drop(popped);
        assert_eq!(drops.get(), 3);
        assert_eq!(stack.last().map(|tracked| tracked.value), Some(20));
        assert_eq!(
            (
                stack.len(),
                stack.is_empty(),
                stack.has_space(),
                stack.is_full(),
            ),
            (2, false, true, false)
        );
        assert_eq!(
            stack
                .as_slice()
                .iter()
                .map(|tracked| tracked.value)
                .collect::<Vec<_>>(),
            [10, 20]
        );

        drop(stack);
        assert_eq!(drops.get(), 5);

        let reused = Stack::<SmokeTracked, _>::try_new_in(3, &block)
            .expect("normal destruction releases the allocation");
        drop(reused);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn into_vec_moves_values_in_order_and_preserves_capacity() {
        let drops = Rc::new(Cell::new(0));
        let block = TypedBlock::<SmokeTracked, 4>::new();
        let mut stack = Stack::try_new_in(4, &block).expect("four values fit");
        for value in [10, 20, 30] {
            assert!(stack.try_push(SmokeTracked::new(value, &drops)).is_none());
        }

        let vec = stack.into_vec();

        assert_eq!(
            vec.iter().map(|tracked| tracked.value).collect::<Vec<_>>(),
            [10, 20, 30]
        );
        assert!(vec.capacity() >= 4);
        assert_eq!(drops.get(), 0);

        let reused = Stack::<SmokeTracked, _>::try_new_in(4, &block)
            .expect("conversion releases the borrowed allocation");
        drop(reused);

        drop(vec);
        assert_eq!(drops.get(), 3);
    }

    #[test]
    fn clone_from_slice_returns_remainder_and_cross_allocator_clone_is_independent() {
        let block = TypedBlock::<String, 3>::new();
        let clone_block = TypedBlock::<String, 3>::new();
        let mut stack = Stack::try_new_in(3, &block).expect("three strings fit");
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

        let cloned = stack
            .try_clone_into(&clone_block)
            .expect("the second block grants a disjoint lease");
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
        let block = TypedBlock::<PanickingClone, 3>::new();
        let mut stack = Stack::try_new_in(3, &block).expect("three values fit");
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

        let block = TypedBlock::<Stored, 4>::new();
        let mut stack = Stack::try_new_in(4, &block).expect("four values fit");
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
    fn slice_trait_forwarding_uses_only_the_initialized_prefix() {
        let left_block = TypedBlock::<u8, 4>::new();
        let right_block = TypedBlock::<u8, 4>::new();
        let mut left = Stack::try_new_in(4, &left_block).expect("four values fit");
        let mut right = Stack::try_new_in(4, &right_block).expect("four values fit");
        assert!(left.as_slice().is_empty());
        left.copy_from_slice(&[1, 2, 3]);
        right.copy_from_slice(&[1, 2, 3]);

        assert_eq!(left.capacity(), 4);
        assert_eq!(left.as_slice(), [1, 2, 3]);
        left.as_mut_slice().swap(0, 2);
        assert_eq!(left.as_slice(), [3, 2, 1]);
        left.as_mut_slice().swap(0, 2);

        assert!(left == right);
        assert_eq!(left[1], 2);
        left[1] = 4;
        assert!(left > right);

        let shared: &[u8] = AsRef::<[u8]>::as_ref(&left);
        assert_eq!(shared, [1, 4, 3]);
        assert_eq!(Borrow::<[u8]>::borrow(&left), [1, 4, 3]);
        assert_eq!(left.first(), Some(&1));

        for value in &mut left {
            *value += 1;
        }
        AsMut::<[u8]>::as_mut(&mut left)[0] = 9;
        BorrowMut::<[u8]>::borrow_mut(&mut left)[2] = 7;
        assert_eq!(left.iter().copied().collect::<Vec<_>>(), [9, 5, 7]);

        right.clear();
        right.copy_from_slice(&[9, 5, 7]);
        let mut left_hash = DefaultHasher::new();
        let mut right_hash = DefaultHasher::new();
        left.hash(&mut left_hash);
        right.hash(&mut right_hash);
        assert_eq!(left_hash.finish(), right_hash.finish());
        assert!(left == right);
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
        let block = TypedBlock::<PanicAfterOneClone, 3>::new();
        let mut stack = Stack::try_new_in(3, &block).expect("three values fit");
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
        let block = TypedBlock::<PanickingDrop, 3>::new();
        let mut stack = Stack::try_new_in(3, &block).expect("three values fit");
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
}
