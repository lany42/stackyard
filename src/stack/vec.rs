// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
//! Growable allocator-backed last-in, first-out collections.
//!
//! [`Vector`] borrows an [`Alloc`] and exposes its initialized values as a
//! bottom-to-top slice. When a push or slice extension exceeds the current
//! capacity, it attempts to grow the live allocation.
//!
//! ```rust
//! use stackyard::{TypedBlock, Vector};
//!
//! let block = TypedBlock::<u8, 4>::new();
//! let mut vector = Vector::<u8, _>::try_new_in(2, &block).unwrap();
//! vector.push(1);
//! vector.push(2);
//! vector.push(3);
//! assert_eq!(vector.pop(), Some(3));
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

/// A growable last-in, first-out collection backed by a borrowed allocator.
///
/// The vector obtains one positive-sized allocation during construction and
/// retains a live lease until destruction. Its initialized values form a
/// bottom-to-top prefix that can be accessed through slice operations. Pushes
/// and slice extensions use the allocator to grow that lease when necessary.
///
/// [`Vector::try_push`] returns a rejected value when the allocation cannot
/// grow. [`Vector::push`] instead silently drops that value. Dropping the
/// vector destroys its initialized values and releases its current allocation.
///
/// Zero capacity and zero-sized element types are unsupported because
/// [`Alloc`] rejects zero-sized layouts.
pub struct Vector<'a, T, A: Alloc + ?Sized> {
    allocator: &'a A,
    ptr: NonNull<MaybeUninit<T>>,
    top: usize,
    capacity: usize,
    _owns: PhantomData<T>,
}

impl<'a, T, A: Alloc + ?Sized> Vector<'a, T, A> {
    /// Attempts to allocate an empty vector with `capacity` slots.
    ///
    /// The vector borrows `allocator` until it is dropped. Returns `None` if the
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

    /// Attempts the amortized growth needed for one additional value.
    fn grow_one(&mut self) -> bool {
        debug_assert!(self.is_full());

        let old_capacity = self.capacity;
        let Some(required_capacity) = old_capacity.checked_add(1) else {
            return false;
        };
        let Some(doubled_capacity) = old_capacity.checked_mul(2) else {
            return false;
        };
        let new_capacity = doubled_capacity.max(required_capacity);
        let Some(new_layout) = Layout::array::<T>(new_capacity).ok() else {
            return false;
        };
        let old_layout = Layout::array::<T>(old_capacity)
            .expect("Vector capacity produced a valid layout during construction or growth");

        // SAFETY:
        // - `self.ptr` identifies this allocator's one live lease
        // - `old_layout` is its exact current positive-sized layout
        // - `new_layout` has the same alignment and is strictly larger
        // - no references into the allocation are live across this call
        let Some(new_ptr) =
            (unsafe { self.allocator.grow(self.ptr.cast(), old_layout, new_layout) })
        else {
            return false;
        };

        self.ptr = new_ptr.cast();
        self.capacity = new_capacity;
        true
    }

    /// Attempts one amortized growth large enough for an entire slice.
    fn grow_for_slice(&mut self, additional: usize) -> bool {
        debug_assert!(additional > self.capacity - self.top);

        let Some(required_capacity) = self.top.checked_add(additional) else {
            return false;
        };
        let old_capacity = self.capacity;
        let Some(doubled_capacity) = old_capacity.checked_mul(2) else {
            return false;
        };
        let new_capacity = doubled_capacity.max(required_capacity);
        let Some(new_layout) = Layout::array::<T>(new_capacity).ok() else {
            return false;
        };
        let old_layout = Layout::array::<T>(old_capacity)
            .expect("Vector capacity produced a valid layout during construction or growth");

        // SAFETY:
        // - `self.ptr` identifies this allocator's one live lease
        // - `old_layout` is its exact current positive-sized layout
        // - `new_layout` has the same alignment and is strictly larger
        // - no references into the allocation are live across this call
        let Some(new_ptr) =
            (unsafe { self.allocator.grow(self.ptr.cast(), old_layout, new_layout) })
        else {
            return false;
        };

        self.ptr = new_ptr.cast();
        self.capacity = new_capacity;
        true
    }

    /// Moves this vector handle into a [`Box`](crate::rust_alloc::boxed::Box).
    ///
    /// The backing allocation remains owned by `allocator`; boxing moves only
    /// the handle.
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn into_boxed(self) -> crate::rust_alloc::boxed::Box<Self> {
        crate::rust_alloc::boxed::Box::new(self)
    }

    /// Moves the vector's values into a freshly allocated
    /// [`Vec`](crate::rust_alloc::vec::Vec).
    ///
    /// The vector is allocated with capacity for every vector slot and
    /// preserves the values' bottom-to-top order. The vector's borrowed raw
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
        // the emptied Vector releases only its raw allocation.
        self.top = 0;
        vec
    }

    /// Returns the current number of values this vector can hold without growing.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of values currently in the vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.top
    }

    /// Returns `true` if the vector contains no values.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.top == 0
    }

    /// Returns `true` if the vector can accept another value without growing.
    #[inline]
    pub fn has_space(&self) -> bool {
        self.top < self.capacity
    }

    /// Returns `true` if the vector would need to grow before accepting a value.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.top == self.capacity
    }

    /// Attempts to push `value` onto the top of the vector.
    ///
    /// Returns `None` when `value` is stored. If the vector is full, first
    /// attempts to grow its allocation. Returns `Some(value)` and leaves the
    /// vector unchanged if growth fails.
    #[inline]
    pub fn try_push(&mut self, value: T) -> Option<T> {
        let top = self.top;

        if top == self.capacity && !self.grow_one() {
            return Some(value);
        }

        // SAFETY:
        // - either `top` was below capacity or `grow_one` enlarged the lease
        // - slots below `top` are initialized and this slot is not
        // - `&mut self` grants exclusive access through the allocation
        unsafe {
            let end = self.ptr.as_ptr().add(top).cast::<T>();
            ptr::write(end, value);
            self.top = top + 1;
        }
        None
    }

    /// Pushes `value` onto the vector, growing its allocation when necessary.
    ///
    /// If required growth fails, the vector remains unchanged and `value` is
    /// silently dropped.
    ///
    /// # Panics
    ///
    /// Panics if growth fails and `value`'s destructor panics while discarding
    /// it. The vector itself remains unchanged.
    #[inline]
    pub fn push(&mut self, value: T) {
        let top = self.top;

        if top == self.capacity && !self.grow_one() {
            return;
        }

        // SAFETY:
        // - either `top` was below capacity or `grow_one` enlarged the lease
        // - slots below `top` are initialized and this slot is not
        // - `&mut self` grants exclusive access through the allocation
        unsafe {
            let end = self.ptr.as_ptr().add(top).cast::<T>();
            ptr::write(end, value);
            self.top = top + 1;
        }
    }

    /// Removes and returns the top value, or `None` if the vector is empty.
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
    /// Returns `None` when the vector is empty.
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

    /// Clones `value` onto the vector until it reaches its current capacity.
    ///
    /// Existing values remain in place. This operation does not grow the
    /// allocation.
    ///
    /// # Panics
    ///
    /// Panics if [`T::clone`](Clone::clone) panics. Clones pushed before the
    /// panic remain in the vector.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        while self.has_space() {
            self.push(value.clone());
        }
    }

    /// Removes and drops all values in the vector.
    ///
    /// # Panics
    ///
    /// Panics if a stored value's destructor panics. The vector is marked empty
    /// before values are dropped.
    #[inline]
    pub fn clear(&mut self) {
        let elements: *mut [T] = self.as_mut_slice();

        // SAFETY:
        // - `elements` is exactly the initialized prefix
        // - resetting `top` first transfers responsibility for that prefix to
        //   this drop operation and prevents any later Vector drop from retrying
        // - compiler-generated slice drop glue handles element destruction
        unsafe {
            self.top = 0;
            ptr::drop_in_place(elements);
        }
    }

    /// Copies all values from `src` onto the vector.
    ///
    /// If the values do not fit, first attempts one amortized growth large
    /// enough for the whole slice. Returns `Some(src)` and leaves the vector
    /// unchanged if growth fails. Returns `None` for an empty or fully copied
    /// slice.
    pub fn copy_from_slice<'b>(&mut self, src: &'b [T]) -> Option<&'b [T]>
    where
        T: Copy,
    {
        if src.is_empty() {
            return None;
        }

        let current = self.top;
        if src.len() > self.capacity - current && !self.grow_for_slice(src.len()) {
            return Some(src);
        }

        let new_top = current + src.len();

        // SAFETY:
        // - the slice fits either in the old lease or the successfully grown one
        // - the range starts at `top`, so every destination slot is uninitialized
        // - `&mut self` grants exclusive access to the destination range
        let destination =
            unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr().add(current), src.len()) };
        destination.write_copy_of_slice(src);
        self.top = new_top;
        None
    }

    /// Clones all values from `src` onto the vector.
    ///
    /// If the values do not fit, first attempts one amortized growth large
    /// enough for the whole slice. Returns `Some(src)` without cloning and
    /// leaves the vector unchanged if growth fails. Returns `None` for an empty
    /// or fully cloned slice.
    ///
    /// # Panics
    ///
    /// Panics if [`T::clone`](Clone::clone) panics. The vector retains the
    /// contents it held before this call.
    pub fn clone_from_slice<'b>(&mut self, src: &'b [T]) -> Option<&'b [T]>
    where
        T: Clone,
    {
        if src.is_empty() {
            return None;
        }

        let current = self.top;
        if src.len() > self.capacity - current && !self.grow_for_slice(src.len()) {
            return Some(src);
        }

        let new_top = current + src.len();

        // SAFETY:
        // - the slice fits either in the old lease or the successfully grown one
        // - the range starts at `top`, so every destination slot is uninitialized
        // - `&mut self` grants exclusive access to the destination range
        let destination =
            unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr().add(current), src.len()) };

        // If cloning panics, this operation drops its partially cloned prefix
        // and leaves `top` unchanged.
        destination.write_clone_of_slice(src);
        self.top = new_top;
        None
    }

    /// Clones this vector into a lease from `allocator`.
    ///
    /// The destination preserves this vector's capacity and initialized prefix.
    /// Returns `None` without cloning any values if the destination layout
    /// overflows or `allocator` cannot grant a disjoint lease.
    ///
    /// # Panics
    ///
    /// Panics if [`T::clone`](Clone::clone) panics. The partially constructed
    /// destination drops every successful clone and releases its lease while
    /// unwinding; this source vector remains unchanged.
    pub fn try_clone_into<'b, B>(&self, allocator: &'b B) -> Option<Vector<'b, T, B>>
    where
        T: Clone,
        B: Alloc + ?Sized,
    {
        let mut cloned = Vector::try_new_in(self.capacity, allocator)?;
        let remainder = cloned.clone_from_slice(self.as_slice());
        debug_assert!(remainder.is_none());
        Some(cloned)
    }
}

impl<'a, T: Hash, A: Alloc + ?Sized> Hash for Vector<'a, T, A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<'a, 'b, T: PartialOrd, A: Alloc + ?Sized, B: Alloc + ?Sized> PartialOrd<Vector<'b, T, B>>
    for Vector<'a, T, A>
{
    #[inline]
    fn partial_cmp(&self, other: &Vector<'b, T, B>) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<'a, T: Ord, A: Alloc + ?Sized> Ord for Vector<'a, T, A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<'a, 'b, T, U, A: Alloc + ?Sized, B: Alloc + ?Sized> PartialEq<Vector<'b, U, B>>
    for Vector<'a, T, A>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &Vector<'b, U, B>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a, T, U, A: Alloc + ?Sized> PartialEq<[U]> for Vector<'a, T, A>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U]) -> bool {
        self.as_slice().eq(other)
    }
}

impl<'a, T, U, A: Alloc + ?Sized> PartialEq<&[U]> for Vector<'a, T, A>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&[U]) -> bool {
        self.as_slice().eq(*other)
    }
}

impl<'a, T, U, A: Alloc + ?Sized, const N: usize> PartialEq<[U; N]> for Vector<'a, T, A>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<'a, T, U, A: Alloc + ?Sized, const N: usize> PartialEq<&[U; N]> for Vector<'a, T, A>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &&[U; N]) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<'a, T: Eq, A: Alloc + ?Sized> Eq for Vector<'a, T, A> {}

impl<'a, T, A: Alloc + ?Sized, I> Index<I> for Vector<'a, T, A>
where
    I: SliceIndex<[T]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self.as_slice(), index)
    }
}

impl<'a, T, A: Alloc + ?Sized, I> IndexMut<I> for Vector<'a, T, A>
where
    I: SliceIndex<[T]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(self.as_mut_slice(), index)
    }
}

impl<'a, T, A: Alloc + ?Sized> AsRef<[T]> for Vector<'a, T, A> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T, A: Alloc + ?Sized> AsMut<[T]> for Vector<'a, T, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T, A: Alloc + ?Sized> AsRef<Vector<'a, T, A>> for Vector<'a, T, A> {
    #[inline]
    fn as_ref(&self) -> &Vector<'a, T, A> {
        self
    }
}

impl<'a, T, A: Alloc + ?Sized> AsMut<Vector<'a, T, A>> for Vector<'a, T, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut Vector<'a, T, A> {
        self
    }
}

impl<'a, T, A: Alloc + ?Sized> Borrow<[T]> for Vector<'a, T, A> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T, A: Alloc + ?Sized> BorrowMut<[T]> for Vector<'a, T, A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T, A: Alloc + ?Sized> Deref for Vector<'a, T, A> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, T, A: Alloc + ?Sized> DerefMut for Vector<'a, T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'vector, 'alloc, T, A: Alloc + ?Sized> IntoIterator for &'vector Vector<'alloc, T, A> {
    type Item = &'vector T;
    type IntoIter = core::slice::Iter<'vector, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'vector, 'alloc, T, A: Alloc + ?Sized> IntoIterator for &'vector mut Vector<'alloc, T, A> {
    type Item = &'vector mut T;
    type IntoIter = core::slice::IterMut<'vector, T>;

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
        // SAFETY: the guard is created from the Vector's one live lease with
        // its exact layout and is the sole owner of the release obligation.
        unsafe { self.allocator.free(self.ptr, self.layout) };
    }
}

impl<T, A: Alloc + ?Sized> Drop for Vector<'_, T, A> {
    #[inline]
    fn drop(&mut self) {
        let layout = Layout::array::<T>(self.capacity)
            .expect("Vector capacity produced a valid layout during construction or growth");

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

    use super::Vector;

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
    fn constructor_failures_leave_the_allocator_available() {
        struct Zst;

        let block = TypedBlock::<u8, 4>::new();

        assert!(Vector::<u8, _>::try_new_in(0, &block).is_none());
        assert!(Vector::<Zst, _>::try_new_in(1, &block).is_none());
        assert!(Vector::<u16, _>::try_new_in(usize::MAX, &block).is_none());
        assert!(Vector::<u8, _>::try_new_in(5, &block).is_none());

        let vector = Vector::<u8, _>::try_new_in(4, &block)
            .expect("the exact layout remains available after each failure");
        assert_eq!(vector.capacity(), 4);
    }

    #[test]
    fn copy_from_slice_grows_for_the_whole_input_without_cloning() {
        COPY_CLONES.store(0, Ordering::Relaxed);
        let block = TypedBlock::<CountedCopy, 7>::new();
        let mut vector = Vector::try_new_in(2, &block).expect("two values fit initially");
        let input = [
            CountedCopy(10),
            CountedCopy(20),
            CountedCopy(30),
            CountedCopy(40),
            CountedCopy(50),
            CountedCopy(60),
        ];
        assert_eq!(vector.try_push(CountedCopy(0)), None);

        assert_eq!(vector.copy_from_slice(&input), None);
        assert_eq!(vector.capacity(), 7);
        assert_eq!(
            vector.as_slice(),
            &[
                CountedCopy(0),
                CountedCopy(10),
                CountedCopy(20),
                CountedCopy(30),
                CountedCopy(40),
                CountedCopy(50),
                CountedCopy(60),
            ]
        );
        assert_eq!(COPY_CLONES.load(Ordering::Relaxed), 0);

        let rejected = [CountedCopy(70)];
        assert_eq!(vector.copy_from_slice(&rejected), Some(&rejected[..]));
        assert_eq!(vector.len(), 7);
        assert_eq!(COPY_CLONES.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn failed_bulk_copy_leaves_available_slots_and_input_untouched() {
        let block = TypedBlock::<u8, 5>::new();
        let mut vector =
            Vector::<u8, _>::try_new_in(3, &block).expect("three values fit initially");
        let input = [10, 20, 30];
        vector.push(1);

        assert_eq!(vector.copy_from_slice(&input), Some(&input[..]));
        assert_eq!(vector.capacity(), 3);
        assert_eq!(vector.as_slice(), [1]);
    }

    #[test]
    fn try_push_pop_and_drop_transfer_each_value_exactly_once() {
        let drops = Rc::new(Cell::new(0));
        let block = TypedBlock::<SmokeTracked, 5>::new();
        let mut vector =
            Vector::<SmokeTracked, _>::try_new_in(2, &block).expect("two values fit initially");

        assert_eq!(
            (
                vector.capacity(),
                vector.len(),
                vector.is_empty(),
                vector.has_space(),
                vector.is_full(),
            ),
            (2, 0, true, true, false)
        );
        assert!(vector.as_slice().is_empty());
        assert!(vector.last().is_none());

        assert!(vector.try_push(SmokeTracked::new(10, &drops)).is_none());
        vector.push(SmokeTracked::new(20, &drops));
        assert!(vector.try_push(SmokeTracked::new(30, &drops)).is_none());
        assert_eq!(vector.capacity(), 4);
        assert_eq!(
            vector
                .as_slice()
                .iter()
                .map(|tracked| tracked.value)
                .collect::<Vec<_>>(),
            [10, 20, 30]
        );
        assert_eq!(vector.last().map(|tracked| tracked.value), Some(30));
        assert_eq!(
            (
                vector.len(),
                vector.is_empty(),
                vector.has_space(),
                vector.is_full(),
            ),
            (3, false, true, false)
        );

        vector.push(SmokeTracked::new(40, &drops));
        assert!(vector.is_full());

        let rejected = vector
            .try_push(SmokeTracked::new(50, &drops))
            .expect("failed exponential growth returns the rejected value");
        assert_eq!(rejected.value, 50);
        drop(rejected);
        assert_eq!(drops.get(), 1);

        vector.push(SmokeTracked::new(60, &drops));
        assert_eq!(drops.get(), 2);
        assert_eq!(
            vector
                .as_slice()
                .iter()
                .map(|tracked| tracked.value)
                .collect::<Vec<_>>(),
            [10, 20, 30, 40]
        );

        let popped = vector.pop().expect("the vector is not empty");
        assert_eq!(popped.value, 40);
        drop(popped);
        assert_eq!(drops.get(), 3);
        assert_eq!(vector.last().map(|tracked| tracked.value), Some(30));
        assert_eq!(
            (
                vector.len(),
                vector.is_empty(),
                vector.has_space(),
                vector.is_full(),
            ),
            (3, false, true, false)
        );
        assert_eq!(
            vector
                .as_slice()
                .iter()
                .map(|tracked| tracked.value)
                .collect::<Vec<_>>(),
            [10, 20, 30]
        );

        drop(vector);
        assert_eq!(drops.get(), 6);

        let reused = Vector::<SmokeTracked, _>::try_new_in(5, &block)
            .expect("destruction releases the final grown allocation");
        drop(reused);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn into_vec_moves_values_in_order_and_preserves_capacity() {
        let drops = Rc::new(Cell::new(0));
        let block = TypedBlock::<SmokeTracked, 4>::new();
        let mut vector = Vector::try_new_in(2, &block).expect("two values fit initially");
        for value in [10, 20, 30] {
            assert!(vector.try_push(SmokeTracked::new(value, &drops)).is_none());
        }
        assert_eq!(vector.capacity(), 4);

        let vec = vector.into_vec();

        assert_eq!(
            vec.iter().map(|tracked| tracked.value).collect::<Vec<_>>(),
            [10, 20, 30]
        );
        assert!(vec.capacity() >= 4);
        assert_eq!(drops.get(), 0);

        let reused = Vector::<SmokeTracked, _>::try_new_in(4, &block)
            .expect("conversion releases the borrowed allocation");
        drop(reused);

        drop(vec);
        assert_eq!(drops.get(), 3);
    }

    #[test]
    fn clone_from_slice_grows_and_cross_allocator_clone_is_independent() {
        let block = TypedBlock::<String, 6>::new();
        let clone_block = TypedBlock::<String, 6>::new();
        let mut vector = Vector::try_new_in(3, &block).expect("three strings fit initially");
        assert!(vector.try_push(String::from("existing")).is_none());
        let input = [
            String::from("left"),
            String::from("right"),
            String::from("remainder"),
        ];

        assert_eq!(vector.clone_from_slice(&input), None);
        assert_eq!(vector.capacity(), 6);
        assert_eq!(
            vector.as_slice(),
            ["existing", "left", "right", "remainder"]
        );
        assert_eq!(vector.clone_from_slice(&[]), None);
        assert_eq!(vector.clone_from_slice(&input), Some(&input[..]));
        assert_eq!(vector.len(), 4);

        let cloned = vector
            .try_clone_into(&clone_block)
            .expect("the second block grants a disjoint lease");
        vector.as_mut_slice()[0].push('!');
        assert_eq!(
            vector.as_slice(),
            ["existing!", "left", "right", "remainder"]
        );
        assert_eq!(
            cloned.as_slice(),
            ["existing", "left", "right", "remainder"]
        );
    }

    struct CountedClone {
        value: u8,
        clones: Rc<Cell<usize>>,
    }

    impl Clone for CountedClone {
        fn clone(&self) -> Self {
            self.clones.set(self.clones.get() + 1);
            Self {
                value: self.value,
                clones: Rc::clone(&self.clones),
            }
        }
    }

    #[test]
    fn failed_bulk_clone_returns_the_entire_input_without_cloning() {
        let clones = Rc::new(Cell::new(0));
        let block = TypedBlock::<CountedClone, 5>::new();
        let mut vector = Vector::try_new_in(3, &block).expect("three values fit initially");
        vector.push(CountedClone {
            value: 1,
            clones: Rc::clone(&clones),
        });
        let input = [10, 20, 30].map(|value| CountedClone {
            value,
            clones: Rc::clone(&clones),
        });

        let rejected = vector
            .clone_from_slice(&input)
            .expect("growth to six exceeds the five-slot allocator");

        assert!(core::ptr::eq(rejected, input.as_slice()));
        assert_eq!(clones.get(), 0);
        assert_eq!(vector.capacity(), 3);
        assert_eq!(
            vector
                .as_slice()
                .iter()
                .map(|value| value.value)
                .collect::<Vec<_>>(),
            [1]
        );
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
        let mut vector = Vector::try_new_in(1, &block).expect("one value fits initially");
        assert!(
            vector
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

        let result = catch_unwind(AssertUnwindSafe(|| vector.clone_from_slice(&input)));

        assert!(result.is_err());
        assert_eq!(vector.capacity(), 3);
        assert_eq!(vector.len(), 1);
        assert!(!vector.as_slice()[0].panic_on_clone);
        assert_eq!(drops.get(), 1);

        drop(vector);
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
        let mut vector = Vector::try_new_in(4, &block).expect("four values fit");
        for value in [1, 2, 3] {
            assert!(vector.try_push(Stored(value)).is_none());
        }

        let array = [Compared(1), Compared(2), Compared(3)];
        let slice = array.as_slice();

        assert!(vector == *slice);
        assert!(vector == slice);
        assert!(vector == array);
        assert!(vector == &array);

        let different = [Compared(1), Compared(2), Compared(4)];
        assert!(vector != different);
        assert!(vector != &different[..2]);
    }

    #[test]
    fn slice_trait_forwarding_uses_only_the_initialized_prefix() {
        let left_block = TypedBlock::<u8, 4>::new();
        let right_block = TypedBlock::<u8, 4>::new();
        let mut left = Vector::try_new_in(4, &left_block).expect("four values fit");
        let mut right = Vector::try_new_in(4, &right_block).expect("four values fit");
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
    fn fill_stops_at_the_current_capacity_without_growing() {
        let block = TypedBlock::<u8, 8>::new();
        let mut vector = Vector::<u8, _>::try_new_in(2, &block).expect("two values fit initially");

        vector.fill(7);

        assert_eq!(vector.capacity(), 2);
        assert_eq!(vector.as_slice(), [7, 7]);
        assert_eq!(vector.try_push(8), None);
        assert_eq!(vector.capacity(), 4);
        assert_eq!(vector.as_slice(), [7, 7, 8]);
    }

    #[test]
    fn fill_leaves_successful_clones_initialized_when_a_later_clone_panics() {
        let clone_calls = Rc::new(Cell::new(0));
        let drops = Rc::new(Cell::new(0));
        let block = TypedBlock::<PanicAfterOneClone, 3>::new();
        let mut vector = Vector::try_new_in(3, &block).expect("three values fit");
        let value = PanicAfterOneClone {
            clone_calls: Rc::clone(&clone_calls),
            drops: Rc::clone(&drops),
        };

        let result = catch_unwind(AssertUnwindSafe(|| vector.fill(value)));

        assert!(result.is_err());
        assert_eq!(clone_calls.get(), 2);
        assert_eq!(vector.len(), 1);
        assert_eq!(drops.get(), 1);

        drop(vector);
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
        let block = TypedBlock::<PanickingDrop, 4>::new();
        let mut vector = Vector::try_new_in(1, &block).expect("one value fits initially");
        for panic_on_drop in [true, false, false] {
            assert!(
                vector
                    .try_push(PanickingDrop {
                        panic_on_drop,
                        drops: Rc::clone(&drops),
                    })
                    .is_none()
            );
        }
        assert_eq!(vector.capacity(), 4);

        let result = catch_unwind(AssertUnwindSafe(|| vector.clear()));

        assert!(result.is_err());
        assert!(vector.is_empty());
        let dropped_during_clear = drops.get();
        assert!(dropped_during_clear > 0);

        for _ in 0..3 {
            assert!(
                vector
                    .try_push(PanickingDrop {
                        panic_on_drop: false,
                        drops: Rc::clone(&drops),
                    })
                    .is_none()
            );
        }

        drop(vector);
        assert_eq!(drops.get(), dropped_during_clear + 3);
    }
}
