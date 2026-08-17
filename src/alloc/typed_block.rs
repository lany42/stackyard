// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
//! Reusable storage sized and aligned for one element type.
//!
//! [`TypedBlock<T, N>`](TypedBlock) reserves space for `N` values of `T` and
//! implements [`Alloc`]. It grants one positive-sized lease at a time, so a
//! collection must release its lease before the block can be reused.
//! Its backing storage has exactly the space and alignment required for those
//! values. Other element types are valid when their requested size and
//! alignment fit within that storage, but using the same element type for the
//! block and its collection is recommended.
//!
//! ```rust
//! use stackyard::{Stack, TypedBlock};
//!
//! let block = TypedBlock::<u8, 2>::new();
//! let mut stack = Stack::<u8, _>::try_new_in(2, &block).unwrap();
//! stack.push(1);
//! assert_eq!(stack.as_slice(), &[1]);
//! ```

use core::{
    alloc::Layout,
    cell::{Cell, UnsafeCell},
    mem::MaybeUninit,
    ptr::NonNull,
};

use super::{Alloc, sealed};

/// A reusable allocator backed by space for `N` values of type `T`.
///
/// `TypedBlock` grants at most one positive-sized allocation at a time. A
/// released block may be allocated again, but two positive-sized leases can
/// never coexist. Zero-sized layouts are unsupported and return `None`.
///
/// This allocator is intended to back collections whose element type is the
/// same `T`, and that pairing is recommended. Because [`Layout`] erases Rust
/// type identity, `TypedBlock` may also accept another element type whose
/// requested size and alignment fit within the backing array. Every request is
/// validated against that array's byte size and alignment.
///
/// A live positive-sized lease points into this value, so the allocator must
/// not be moved until the lease is released. Successful growth replaces it
/// with another live lease and does not permit moving the allocator. Safe
/// collections enforce this by retaining a shared borrow of the `TypedBlock`.
/// Releasing a lease or dropping the block does not drop values in its storage.
pub struct TypedBlock<T, const N: usize> {
    slots: UnsafeCell<[MaybeUninit<T>; N]>,
    occupied: Cell<bool>,
}

impl<T, const N: usize> TypedBlock<T, N> {
    /// Creates a block with no active lease.
    #[inline]
    pub const fn new() -> Self {
        Self {
            slots: UnsafeCell::new([const { MaybeUninit::uninit() }; N]),
            occupied: Cell::new(false),
        }
    }

    #[inline]
    fn fits(layout: Layout) -> bool {
        let storage = Layout::new::<[MaybeUninit<T>; N]>();
        layout.size() <= storage.size() && layout.align() <= storage.align()
    }

    #[inline]
    fn storage_ptr(&self) -> NonNull<u8> {
        let ptr = self.slots.get().cast::<u8>();

        // SAFETY: `UnsafeCell::get` points to a field within the live `self`,
        // so it cannot be null. No reference to the storage is created.
        unsafe { NonNull::new_unchecked(ptr) }
    }
}

impl<T, const N: usize> Default for TypedBlock<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> sealed::Sealed for TypedBlock<T, N> {}

// SAFETY:
// - `occupied` permits only one positive-sized lease over `slots`
// - every positive-sized request is checked against the complete size and
//   alignment of `slots` before the lease is granted
// - `UnsafeCell` permits the lease holder to initialize and mutate the raw
//   storage while the allocator is held through a shared reference
// - allocator operations touch only `occupied` while a positive lease is live
// - zero-sized requests return `None` without changing allocation state
// - valid release and growth paths cannot unwind after changing state
unsafe impl<T, const N: usize> Alloc for TypedBlock<T, N> {
    #[inline]
    fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return None;
        }

        if self.occupied.get() || !Self::fits(layout) {
            return None;
        }

        let ptr = self.storage_ptr();
        self.occupied.set(true);
        Some(ptr)
    }

    #[inline]
    unsafe fn free(&self, ptr: NonNull<u8>, layout: Layout) {
        debug_assert!(layout.size() > 0);
        debug_assert!(self.occupied.get());
        debug_assert!(Self::fits(layout));
        debug_assert_eq!(ptr, self.storage_ptr());

        self.occupied.set(false);
    }

    #[inline]
    unsafe fn grow(
        &self,
        old_ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Option<NonNull<u8>> {
        debug_assert!(old_layout.size() > 0);
        debug_assert!(new_layout.size() >= old_layout.size());

        debug_assert!(self.occupied.get());
        debug_assert!(Self::fits(old_layout));
        debug_assert_eq!(old_ptr, self.storage_ptr());

        if !Self::fits(new_layout) {
            return None;
        }

        Some(old_ptr)
    }
}

#[cfg(test)]
mod tests {
    use core::{alloc::Layout, cell::Cell};

    use super::{Alloc, TypedBlock};

    #[test]
    fn allocates_an_exact_t_array_layout_at_the_backing_address() {
        let block = TypedBlock::<u32, 4>::new();
        let layout = Layout::array::<u32>(4).unwrap();

        let allocation = block.alloc(layout).expect("the exact layout fits");

        assert_eq!(allocation, block.storage_ptr());
        assert_eq!(allocation.addr().get() % layout.align(), 0);
        assert!(block.occupied.get());

        // SAFETY: `allocation` is the live lease returned for `layout`.
        unsafe { block.free(allocation, layout) };
        assert!(!block.occupied.get());
    }

    #[test]
    fn rejects_oversized_and_overaligned_layouts_without_occupying() {
        let block = TypedBlock::<u8, 64>::new();
        let oversized = Layout::from_size_align(65, 1).unwrap();
        let overaligned = Layout::from_size_align(1, 16).unwrap();

        assert_eq!(block.alloc(oversized), None);
        assert_eq!(block.alloc(overaligned), None);
        assert!(!block.occupied.get());

        let exact = Layout::array::<u8>(64).unwrap();
        let allocation = block.alloc(exact).expect("failures leave the block free");

        // SAFETY: `allocation` is the live lease returned for `exact`.
        unsafe { block.free(allocation, exact) };
    }

    #[test]
    fn rejects_a_second_positive_lease_without_touching_the_first() {
        let block = TypedBlock::<u32, 4>::new();
        let layout = Layout::array::<u32>(2).unwrap();
        let allocation = block.alloc(layout).expect("the first request fits");
        let words = allocation.cast::<u32>().as_ptr();

        // SAFETY: the lease covers two aligned `u32` slots, and the caller has
        // exclusive authority over both.
        unsafe {
            words.write(0x1234_5678);
            words.add(1).write(0x90ab_cdef);
        }

        assert_eq!(block.alloc(Layout::new::<u32>()), None);
        assert!(block.occupied.get());

        // SAFETY: failed allocation leaves the original lease and its bytes
        // unchanged. `u32` is `Copy`, so these reads do not move ownership.
        unsafe {
            assert_eq!(words.read(), 0x1234_5678);
            assert_eq!(words.add(1).read(), 0x90ab_cdef);
            block.free(allocation, layout);
        }
    }

    #[test]
    fn rejects_zero_sized_layouts_without_occupying() {
        struct Zst;

        let block = TypedBlock::<u32, 4>::new();
        let align_one = Layout::from_size_align(0, 1).unwrap();
        let align_sixty_four = Layout::from_size_align(0, 64).unwrap();

        assert_eq!(block.alloc(align_one), None);
        assert_eq!(block.alloc(align_sixty_four), None);
        assert!(!block.occupied.get());

        let positive_layout = Layout::new::<u32>();
        let positive = block
            .alloc(positive_layout)
            .expect("zero-sized failures leave the block available");

        // SAFETY: the positive lease covers one aligned `u32`.
        unsafe { positive.cast::<u32>().as_ptr().write(123) };

        assert_eq!(block.alloc(align_sixty_four), None);
        assert!(block.occupied.get());

        // SAFETY: the rejected zero-sized request leaves the positive lease
        // and its initialized contents unchanged.
        unsafe {
            assert_eq!(positive.cast::<u32>().as_ptr().read(), 123);
            block.free(positive, positive_layout);
        }

        let zst_block = TypedBlock::<Zst, 8>::new();
        let zst_layout = Layout::array::<Zst>(8).unwrap();

        assert_eq!(zst_block.alloc(zst_layout), None);
        assert_eq!(zst_block.alloc(Layout::new::<u8>()), None);
        assert!(!zst_block.occupied.get());
    }

    #[test]
    fn release_permits_sequential_reuse_at_the_same_address() {
        let block = TypedBlock::<u64, 2>::new();
        let layout = Layout::array::<u64>(2).unwrap();
        let first = block.alloc(layout).expect("the first request fits");
        let address = first.addr();

        // SAFETY: `first` is the live lease returned for `layout`.
        unsafe { block.free(first, layout) };

        let second = block.alloc(layout).expect("the released block is reusable");
        assert_eq!(second.addr(), address);

        // SAFETY: `second` is the current live lease for `layout`.
        unsafe { block.free(second, layout) };
    }

    #[test]
    fn grow_extends_a_positive_lease_in_place_and_preserves_bytes() {
        let block = TypedBlock::<u32, 4>::new();
        let old_layout = Layout::array::<u32>(2).unwrap();
        let new_layout = Layout::array::<u32>(4).unwrap();
        let old_ptr = block.alloc(old_layout).expect("the old layout fits");
        let old_address = old_ptr.addr();
        let words = old_ptr.cast::<u32>().as_ptr();

        // SAFETY: the old lease covers two aligned `u32` slots.
        unsafe {
            words.write(10);
            words.add(1).write(20);
        }

        // SAFETY: `old_ptr` is the live lease for `old_layout`, the layout is
        // unchanged, and no references into the storage are live.
        let current = unsafe { block.grow(old_ptr, old_layout, old_layout) }
            .expect("identical-layout growth always succeeds");

        // SAFETY: `current` is now the live lease for `old_layout`, the new
        // layout is larger, and no references into the storage are live.
        let grown = unsafe { block.grow(current, old_layout, new_layout) }
            .expect("the backing storage has room for in-place growth");

        assert_eq!(grown.addr(), old_address);
        assert!(block.occupied.get());
        assert_eq!(block.alloc(Layout::new::<u32>()), None);

        let grown_words = grown.cast::<u32>().as_ptr();
        // SAFETY: growth preserved the first two slots and extended the lease
        // to cover all four aligned slots.
        unsafe {
            assert_eq!(grown_words.read(), 10);
            assert_eq!(grown_words.add(1).read(), 20);
            grown_words.add(2).write(30);
            grown_words.add(3).write(40);
            assert_eq!(grown_words.add(2).read(), 30);
            assert_eq!(grown_words.add(3).read(), 40);
            block.free(grown, new_layout);
        }
    }

    #[test]
    fn failed_positive_growth_is_transactional() {
        let block = TypedBlock::<u32, 4>::new();
        let old_layout = Layout::array::<u32>(2).unwrap();
        let too_large = Layout::array::<u32>(5).unwrap();
        let too_aligned = Layout::from_size_align(old_layout.size(), 8).unwrap();
        let old_ptr = block.alloc(old_layout).expect("the old layout fits");
        let words = old_ptr.cast::<u32>().as_ptr();

        // SAFETY: the old lease covers two aligned `u32` slots.
        unsafe {
            words.write(41);
            words.add(1).write(42);
        }

        // SAFETY: each call receives the current lease and a non-smaller new
        // layout. A `None` result restores that lease to the caller.
        assert_eq!(unsafe { block.grow(old_ptr, old_layout, too_large) }, None);
        assert_eq!(
            unsafe { block.grow(old_ptr, old_layout, too_aligned) },
            None
        );

        assert!(block.occupied.get());
        assert_eq!(block.alloc(Layout::new::<u32>()), None);

        // SAFETY: both failed growth attempts retained the old lease and its
        // initialized contents.
        unsafe {
            assert_eq!(words.read(), 41);
            assert_eq!(words.add(1).read(), 42);
            block.free(old_ptr, old_layout);
        }
    }

    #[test]
    fn grow_inner_refuses_to_overlap_an_occupied_positive_block() {
        let block = TypedBlock::<u32, 4>::new();
        let old_layout = Layout::array::<u32>(1).unwrap();
        let new_layout = Layout::array::<u32>(4).unwrap();
        let old_ptr = block.alloc(old_layout).expect("the old layout fits");

        // SAFETY: the lease covers one aligned `u32` slot.
        unsafe { old_ptr.cast::<u32>().as_ptr().write(99) };

        // SAFETY: `old_ptr` is the live lease and the layouts satisfy the
        // growth-size precondition. The fallback must allocate before freeing.
        let current = unsafe { block.grow_inner(old_ptr, old_layout, old_layout) }
            .expect("identical layouts return the current handle");
        // SAFETY: `current` is now the live lease. A failed fallback restores
        // it to the caller unchanged.
        assert_eq!(
            unsafe { block.grow_inner(current, old_layout, new_layout) },
            None
        );

        assert!(block.occupied.get());
        // SAFETY: failed growth retained the current lease and its contents.
        unsafe {
            assert_eq!(current.cast::<u32>().as_ptr().read(), 99);
            block.free(current, old_layout);
        }
    }

    #[test]
    fn allocator_never_drops_values_in_released_storage() {
        struct Tracked<'a>(&'a Cell<usize>);

        impl Drop for Tracked<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let drops = Cell::new(0);
        {
            let block = TypedBlock::<Tracked<'_>, 1>::new();
            let layout = Layout::new::<Tracked<'_>>();
            let allocation = block.alloc(layout).expect("the exact layout fits");

            // SAFETY: the lease is correctly aligned and large enough for one
            // `Tracked`. Ownership of the value is deliberately leaked to
            // verify that releasing raw storage does not run destruction.
            unsafe {
                allocation
                    .cast::<Tracked<'_>>()
                    .as_ptr()
                    .write(Tracked(&drops));
                block.free(allocation, layout);
            }

            assert_eq!(drops.get(), 0);
        }

        assert_eq!(drops.get(), 0);
    }

    #[test]
    fn positive_growth_does_not_read_uninitialized_storage() {
        let block = TypedBlock::<u64, 4>::new();
        let old_layout = Layout::array::<u64>(1).unwrap();
        let new_layout = Layout::array::<u64>(4).unwrap();
        let old_ptr = block.alloc(old_layout).expect("the old layout fits");

        // SAFETY: the old lease is live and the larger layout fits. Its bytes
        // intentionally remain uninitialized; in-place growth must not read
        // or copy them.
        let grown = unsafe { block.grow(old_ptr, old_layout, new_layout) }
            .expect("in-place growth does not inspect stored bytes");

        // SAFETY: `grown` is the current live lease for `new_layout`.
        unsafe { block.free(grown, new_layout) };
    }
}
