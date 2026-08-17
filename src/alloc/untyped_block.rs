// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
//! Untyped single-block allocation.

use core::{
    alloc::Layout,
    cell::{Cell, UnsafeCell},
    mem::MaybeUninit,
    ptr::NonNull,
};

use super::{Alloc, sealed};

/// A reusable allocator backed by `N` bytes of untyped storage.
///
/// `UntypedBlock` grants at most one positive-sized allocation at a time. A
/// released block may be allocated again for the same layout or a different
/// one, but two positive-sized leases can never coexist. Zero-sized layouts
/// are unsupported and return `None`.
///
/// The byte array has alignment one. For each request, the allocator derives
/// the first suitably aligned interior pointer from the array's actual
/// address, then verifies that the complete requested range remains within
/// the `N` backing bytes. Consequently, a request can fail because of leading
/// alignment padding even when `layout.size() <= N`.
///
/// A live positive-sized lease points into this value. The allocator must not
/// be moved until that lease has been released or successfully grown. Safe
/// collections enforce this by retaining a shared borrow of the
/// `UntypedBlock`.
pub struct UntypedBlock<const N: usize> {
    bytes: UnsafeCell<[MaybeUninit<u8>; N]>,
    occupied: Cell<bool>,
}

impl<const N: usize> UntypedBlock<N> {
    /// Creates an unoccupied block.
    #[inline]
    pub const fn new() -> Self {
        Self {
            bytes: UnsafeCell::new([MaybeUninit::uninit(); N]),
            occupied: Cell::new(false),
        }
    }

    #[inline]
    fn storage_ptr(&self) -> NonNull<u8> {
        let ptr = self.bytes.get().cast::<u8>();

        // SAFETY: `UnsafeCell::get` points to a field within the live `self`,
        // so it cannot be null. No reference to the storage is created.
        unsafe { NonNull::new_unchecked(ptr) }
    }

    #[inline]
    fn fitting_ptr(&self, layout: Layout) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return None;
        }

        let base = self.storage_ptr();
        let padding = base.as_ptr().align_offset(layout.align());
        if padding == usize::MAX {
            return None;
        }

        let end = padding.checked_add(layout.size())?;

        if end > N {
            return None;
        }

        // SAFETY: a positive `layout.size()` and `end <= N` imply
        // `padding < N`, so this pointer remains within the backing byte
        // array. Deriving it with `add` preserves the backing allocation's
        // provenance. `align_offset` establishes the requested alignment.
        let ptr = unsafe { base.as_ptr().add(padding) };

        // SAFETY: `ptr` is derived without wrapping from the non-null `base`
        // and remains within the same live allocation.
        Some(unsafe { NonNull::new_unchecked(ptr) })
    }
}

impl<const N: usize> Default for UntypedBlock<N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> sealed::Sealed for UntypedBlock<N> {}

// SAFETY:
// - `occupied` permits only one positive-sized lease over `bytes`
// - every positive-sized request is dynamically aligned within `bytes` and
//   accepted only when its complete range fits
// - every returned pointer is derived from the byte-array pointer, preserving
//   provenance without relying on the enclosing struct's field layout
// - `UnsafeCell` permits the lease holder to initialize and mutate the raw
//   storage while the allocator is held through a shared reference
// - allocator operations touch only `occupied` while a positive lease is live
// - zero-sized requests return `None` without changing allocation state
// - valid release and growth paths cannot unwind after changing state
unsafe impl<const N: usize> Alloc for UntypedBlock<N> {
    #[inline]
    fn alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return None;
        }

        if self.occupied.get() {
            return None;
        }

        let ptr = self.fitting_ptr(layout)?;
        self.occupied.set(true);
        Some(ptr)
    }

    #[inline]
    unsafe fn free(&self, ptr: NonNull<u8>, layout: Layout) {
        debug_assert!(layout.size() > 0);
        debug_assert!(self.occupied.get());
        debug_assert_eq!(self.fitting_ptr(layout), Some(ptr));

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
        debug_assert_eq!(self.fitting_ptr(old_layout), Some(old_ptr));

        let new_ptr = self.fitting_ptr(new_layout)?;
        if new_ptr != old_ptr {
            return None;
        }

        Some(old_ptr)
    }
}

#[cfg(test)]
mod tests {
    use core::{alloc::Layout, cell::Cell};

    use super::{Alloc, UntypedBlock};

    fn block_with_odd_storage(blocks: &[UntypedBlock<64>; 2]) -> &UntypedBlock<64> {
        blocks
            .iter()
            .find(|block| block.storage_ptr().addr().get() % 2 == 1)
            .expect("an odd-sized block array places one byte array at an odd address")
    }

    #[test]
    fn allocates_an_alignment_one_layout_at_the_backing_address() {
        let block = UntypedBlock::<16>::new();
        let layout = Layout::from_size_align(16, 1).unwrap();

        let allocation = block.alloc(layout).expect("the exact byte range fits");

        assert_eq!(allocation, block.storage_ptr());
        assert!(block.occupied.get());

        // SAFETY: `allocation` is the live lease returned for `layout`.
        unsafe { block.free(allocation, layout) };
        assert!(!block.occupied.get());
    }

    #[test]
    fn aligns_an_interior_pointer_with_runtime_padding() {
        let blocks = [UntypedBlock::<64>::new(), UntypedBlock::<64>::new()];
        let block = block_with_odd_storage(&blocks);
        let base_addr = block.storage_ptr().addr().get();
        let layout = Layout::from_size_align(7, 2).unwrap();

        let allocation = block.alloc(layout).expect("one byte of padding fits");

        assert_eq!(allocation.addr().get(), base_addr + 1);
        assert_eq!(allocation.addr().get() % layout.align(), 0);

        // SAFETY: the lease covers seven writable bytes. Every byte is
        // initialized before it is read.
        unsafe {
            for index in 0..layout.size() {
                allocation.as_ptr().add(index).write(index as u8);
            }
            for index in 0..layout.size() {
                assert_eq!(allocation.as_ptr().add(index).read(), index as u8);
            }
            block.free(allocation, layout);
        }
    }

    #[test]
    fn rejects_a_range_that_fails_only_after_padding_is_included() {
        let blocks = [UntypedBlock::<64>::new(), UntypedBlock::<64>::new()];
        let block = block_with_odd_storage(&blocks);
        let too_large_after_padding = Layout::from_size_align(64, 2).unwrap();

        assert_eq!(block.alloc(too_large_after_padding), None);
        assert!(!block.occupied.get());

        let exact_after_padding = Layout::from_size_align(63, 2).unwrap();
        let allocation = block
            .alloc(exact_after_padding)
            .expect("one padding byte plus 63 requested bytes fits exactly");

        // SAFETY: `allocation` is the live lease for `exact_after_padding`.
        unsafe { block.free(allocation, exact_after_padding) };
    }

    #[test]
    fn rejects_a_second_positive_lease_without_touching_the_first() {
        let block = UntypedBlock::<32>::new();
        let layout = Layout::new::<u64>();
        let allocation = block.alloc(layout).expect("one aligned u64 fits");
        let word = allocation.cast::<u64>().as_ptr();

        // SAFETY: the lease covers one aligned `u64`.
        unsafe { word.write(0x1234_5678_90ab_cdef) };

        assert_eq!(block.alloc(Layout::new::<u8>()), None);
        assert!(block.occupied.get());

        // SAFETY: failed allocation leaves the original lease and its bytes
        // unchanged. `u64` is `Copy`, so the read does not move ownership.
        unsafe {
            assert_eq!(word.read(), 0x1234_5678_90ab_cdef);
            block.free(allocation, layout);
        }
    }

    #[test]
    fn rejects_zero_sized_layouts_without_occupying() {
        let block = UntypedBlock::<16>::new();
        let align_one = Layout::from_size_align(0, 1).unwrap();
        let align_sixty_four = Layout::from_size_align(0, 64).unwrap();

        assert_eq!(block.alloc(align_one), None);
        assert_eq!(block.alloc(align_sixty_four), None);
        assert!(!block.occupied.get());

        let positive_layout = Layout::new::<u8>();
        let positive = block
            .alloc(positive_layout)
            .expect("zero-sized failures leave the block available");

        // SAFETY: the positive lease covers one byte.
        unsafe { positive.as_ptr().write(123) };

        assert_eq!(block.alloc(align_sixty_four), None);
        assert!(block.occupied.get());

        // SAFETY: the rejected zero-sized request leaves the positive lease
        // and its initialized byte unchanged.
        unsafe {
            assert_eq!(positive.as_ptr().read(), 123);
            block.free(positive, positive_layout);
        }

        let empty = UntypedBlock::<0>::new();
        assert_eq!(empty.alloc(Layout::new::<u8>()), None);
        assert_eq!(empty.alloc(align_one), None);
        assert!(!empty.occupied.get());
    }

    #[test]
    fn reuses_the_block_sequentially_for_different_element_types() {
        let block = UntypedBlock::<64>::new();
        let shorts_layout = Layout::array::<u16>(3).unwrap();
        let shorts = block
            .alloc(shorts_layout)
            .expect("three aligned u16 values fit");
        let short_ptr = shorts.cast::<u16>().as_ptr();

        // SAFETY: the lease covers three aligned `u16` values.
        unsafe {
            short_ptr.write(10);
            short_ptr.add(1).write(20);
            short_ptr.add(2).write(30);
            assert_eq!(short_ptr.add(1).read(), 20);
            block.free(shorts, shorts_layout);
        }

        let longs_layout = Layout::array::<u64>(2).unwrap();
        let longs = block
            .alloc(longs_layout)
            .expect("the released bytes can hold two aligned u64 values");
        let long_ptr = longs.cast::<u64>().as_ptr();

        // SAFETY: the current lease covers two aligned `u64` values.
        unsafe {
            long_ptr.write(100);
            long_ptr.add(1).write(200);
            assert_eq!(long_ptr.read(), 100);
            assert_eq!(long_ptr.add(1).read(), 200);
            block.free(longs, longs_layout);
        }
    }

    #[test]
    fn grow_extends_a_positive_lease_in_place_and_preserves_bytes() {
        let block = UntypedBlock::<64>::new();
        let old_layout = Layout::array::<u32>(2).unwrap();
        let new_layout = Layout::array::<u32>(6).unwrap();
        let old_ptr = block.alloc(old_layout).expect("the old layout fits");
        let old_address = old_ptr.addr();
        let words = old_ptr.cast::<u32>().as_ptr();

        // SAFETY: the old lease covers two aligned `u32` values.
        unsafe {
            words.write(10);
            words.add(1).write(20);
        }

        // SAFETY: `old_ptr` is the live lease for `old_layout`, and no
        // references into the storage are live.
        let current = unsafe { block.grow(old_ptr, old_layout, old_layout) }
            .expect("identical-layout growth always succeeds");

        // SAFETY: `current` is now the live lease, the new layout is larger,
        // and no references into the storage are live.
        let grown = unsafe { block.grow(current, old_layout, new_layout) }
            .expect("the larger range fits at the same aligned address");

        assert_eq!(grown.addr(), old_address);
        assert!(block.occupied.get());
        assert_eq!(block.alloc(Layout::new::<u8>()), None);

        let grown_words = grown.cast::<u32>().as_ptr();
        // SAFETY: growth preserved the first two values and extended the
        // writable lease to all six aligned slots.
        unsafe {
            assert_eq!(grown_words.read(), 10);
            assert_eq!(grown_words.add(1).read(), 20);
            grown_words.add(5).write(60);
            assert_eq!(grown_words.add(5).read(), 60);
            block.free(grown, new_layout);
        }
    }

    #[test]
    fn failed_growth_is_transactional() {
        let blocks = [UntypedBlock::<64>::new(), UntypedBlock::<64>::new()];
        let block = block_with_odd_storage(&blocks);
        let old_layout = Layout::from_size_align(8, 1).unwrap();
        let relocated_layout = Layout::from_size_align(16, 2).unwrap();
        let oversized_layout = Layout::from_size_align(65, 1).unwrap();
        let old_ptr = block.alloc(old_layout).expect("the old layout fits");

        // SAFETY: the old lease covers eight writable bytes.
        unsafe {
            for index in 0..old_layout.size() {
                old_ptr.as_ptr().add(index).write((index + 1) as u8);
            }
        }

        // SAFETY: each call receives the current lease and a non-smaller new
        // layout. A `None` result restores that lease to the caller.
        assert_eq!(
            unsafe { block.grow(old_ptr, old_layout, relocated_layout) },
            None
        );
        assert_eq!(
            unsafe { block.grow(old_ptr, old_layout, oversized_layout) },
            None
        );

        assert!(block.occupied.get());
        assert_eq!(block.alloc(Layout::new::<u8>()), None);

        // SAFETY: both failures retained the old lease and all initialized
        // bytes unchanged.
        unsafe {
            for index in 0..old_layout.size() {
                assert_eq!(old_ptr.as_ptr().add(index).read(), (index + 1) as u8);
            }
            block.free(old_ptr, old_layout);
        }
    }

    #[test]
    fn grow_inner_refuses_to_overlap_an_occupied_positive_block() {
        let block = UntypedBlock::<64>::new();
        let old_layout = Layout::array::<u32>(1).unwrap();
        let new_layout = Layout::array::<u32>(4).unwrap();
        let old_ptr = block.alloc(old_layout).expect("the old layout fits");

        // SAFETY: the lease covers one aligned `u32` value.
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
            let block = UntypedBlock::<64>::new();
            let layout = Layout::new::<Tracked<'_>>();
            let allocation = block.alloc(layout).expect("one Tracked value fits");

            // SAFETY: the lease is aligned and large enough for one `Tracked`.
            // Ownership is deliberately leaked to verify that releasing raw
            // storage does not run destruction.
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
        let block = UntypedBlock::<64>::new();
        let old_layout = Layout::array::<u64>(1).unwrap();
        let new_layout = Layout::array::<u64>(4).unwrap();
        let old_ptr = block.alloc(old_layout).expect("the old layout fits");

        // SAFETY: the old lease is live and the larger layout fits at the same
        // address. Its bytes intentionally remain uninitialized; in-place
        // growth must not read or copy them.
        let grown = unsafe { block.grow(old_ptr, old_layout, new_layout) }
            .expect("in-place growth does not inspect stored bytes");

        // SAFETY: `grown` is the current live lease for `new_layout`.
        unsafe { block.free(grown, new_layout) };
    }
}
