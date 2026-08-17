// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
//! Raw-storage allocation for stackyard collections.
//!
//! [`Alloc`] is public for use in collection bounds, but sealed so allocator
//! implementations remain inside this crate. Implementations grant exclusive
//! leases over raw storage; collections remain responsible for initialization,
//! typed validity, and destruction of values stored there.
//!
//! Zero-sized layouts are unsupported and are rejected without producing a
//! pointer or changing allocator state.

use core::{alloc::Layout, ptr::NonNull};

mod typed_block;
mod untyped_block;

pub use typed_block::TypedBlock;
pub use untyped_block::UntypedBlock;

mod sealed {
    /// Prevents downstream implementations of [`super::Alloc`].
    pub trait Sealed {}
}

/// A crate-provided allocator of exclusive raw-storage leases.
///
/// A successful positive-sized allocation grants its caller exclusive access
/// authority over the requested byte range. Pointer values may be copied, but
/// no other live allocation may grant access to any of the same bytes. This
/// requirement applies across every allocator handle or wrapper that shares
/// the same backing storage.
///
/// While a positive-sized lease is live, neither the allocator nor another
/// client may read or write its bytes, expose a reference to them, relocate
/// them, or invalidate them. The only exceptions are [`Alloc::grow`] and
/// [`Alloc::free`], after their caller has surrendered access as required by
/// those methods.
///
/// # Safety
///
/// Implementations must uphold every lease, layout, ownership-transition, and
/// failure guarantee documented on this trait and its methods. In particular,
/// positive-sized live leases over shared backing storage must always be
/// pairwise disjoint, even when obtained through different allocator handles.
///
/// Ordinary allocation failures return `None`. An implementation may use a
/// diagnostic assertion for a broken invariant only before changing allocator
/// state or calling an operation that may change it. If that assertion
/// unwinds, all allocation metadata, live leases, and leased bytes must remain
/// exactly as they were on method entry. Once a state transition is committed,
/// the remaining valid-call path must not unwind. A valid call to
/// [`Alloc::free`] must not panic.
pub unsafe trait Alloc: sealed::Sealed {
    /// Attempts to allocate a block for `layout`.
    ///
    /// For a positive-sized layout, `Some(ptr)` begins a live lease that:
    ///
    /// - covers at least `layout.size()` bytes starting at `ptr`;
    /// - satisfies `layout.align()`;
    /// - permits raw writes and untyped bytewise copies over the requested
    ///   range, including operations that preserve uninitialized bytes;
    /// - grants the caller exclusive access authority over that range;
    /// - is disjoint from every other positive-sized live lease over the same
    ///   backing storage; and
    /// - remains valid and at a stable address until it is passed to a
    ///   successful [`Alloc::grow`] or to [`Alloc::free`].
    ///
    /// The lease belongs to this allocator instance and must later be passed
    /// back with its exact current layout.
    ///
    /// A zero-sized `layout` is unsupported and must return `None` without
    /// changing allocator state.
    ///
    /// `None` creates no lease and leaves the allocator's state, every live
    /// lease, and all leased bytes unchanged. An implementation must return
    /// `None` rather than a pointer if it cannot establish every applicable
    /// guarantee above. The returned pointer does not state that any byte is
    /// initialized; callers must not read a value or form a reference to one
    /// until they have established its initialization and validity.
    fn alloc(&self, layout: Layout) -> Option<NonNull<u8>>;

    /// Releases a live lease.
    ///
    /// # Safety
    ///
    /// `layout.size()` must be positive. `ptr` must identify a live lease
    /// produced by this allocator instance, and `layout` must be its exact
    /// current layout. The caller must hold the lease's exclusive access
    /// authority, surrender it for this call, release the lease exactly once,
    /// and never access `ptr` or a pointer derived from it afterward.
    ///
    /// A call satisfying these requirements must not panic.
    unsafe fn free(&self, ptr: NonNull<u8>, layout: Layout);

    /// Attempts to replace a live lease with one satisfying `new_layout`.
    ///
    /// Implementations may grow the block in place or call
    /// [`Alloc::grow_inner`] for the allocate-copy-free fallback.
    ///
    /// `None` leaves the allocator's state, the old lease and pointer, every
    /// unrelated lease, and all leased bytes unchanged. It restores the
    /// caller's exclusive authority over the old lease.
    ///
    /// `Some(new_ptr)` begins a replacement lease satisfying every applicable
    /// guarantee of [`Alloc::alloc`] for `new_layout` and preserves the first
    /// `old_layout.size()` bytes. Success consumes the old lease: `old_ptr` and
    /// every pointer derived from it become invalid, even if `new_ptr` has the
    /// same numerical address. The returned pointer carries the sole access
    /// authority for the replacement lease.
    ///
    /// Identical layouts always succeed and may return `old_ptr`.
    ///
    /// # Safety
    ///
    /// `old_ptr` must identify a live lease produced by this allocator
    /// instance, and `old_layout` must be its exact current layout. Both
    /// layouts must have positive size, and `new_layout.size()` must be at
    /// least `old_layout.size()`. The caller must hold the old lease's
    /// exclusive access authority and ensure that no references into the old
    /// block remain live across this call. The caller temporarily surrenders
    /// that authority until the result determines whether the old or
    /// replacement lease is live.
    unsafe fn grow(
        &self,
        old_ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Option<NonNull<u8>>;

    /// Implements growth by allocating, copying, and releasing the old lease.
    ///
    /// Allocators may call this from [`Alloc::grow`] when an in-place
    /// optimization is unavailable. Because the old lease remains live while
    /// [`Alloc::alloc`] obtains the replacement, the implementation of
    /// `alloc` must return a disjoint positive-sized block. Allocation occurs
    /// before release, so allocation failure leaves the old lease untouched.
    ///
    /// Each allocator implementation is responsible for testing this fallback
    /// against its own allocation behavior, including disjoint replacement,
    /// byte preservation, ownership transfer, and transactional failure.
    ///
    /// # Safety
    ///
    /// The caller must satisfy all preconditions of [`Alloc::grow`]. The
    /// result has the same ownership-transition and failure guarantees as
    /// `grow`.
    unsafe fn grow_inner(
        &self,
        old_ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Option<NonNull<u8>> {
        debug_assert!(old_layout.size() > 0);
        debug_assert!(new_layout.size() >= old_layout.size());

        if old_layout == new_layout {
            return Some(old_ptr);
        }

        let new_ptr = self.alloc(new_layout)?;

        // SAFETY:
        // - the caller guarantees that `old_ptr` covers `old_layout.size()`
        //   bytes and has temporarily surrendered exclusive access
        // - `alloc` guarantees that `new_ptr` covers at least that many bytes
        //   because `new_layout` is no smaller than `old_layout`
        // - while the old lease is live, `alloc` must return a disjoint block
        // - `free` receives the still-live old lease and its exact layout
        unsafe {
            core::ptr::copy_nonoverlapping(old_ptr.as_ptr(), new_ptr.as_ptr(), old_layout.size());
            self.free(old_ptr, old_layout);
        }

        Some(new_ptr)
    }
}
