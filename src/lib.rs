// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
//! # stackyard
//!
//! Freestanding data structures and algorithms.
//!
//! The crate currently provides fixed-capacity stacks and reusable storage for
//! `no_std` programs. [`InlineStack`] owns its storage and has a capacity fixed
//! by a const generic. [`Stack`] instead borrows a crate-provided [`Alloc`] and
//! chooses its capacity at run time. [`TypedBlock`] provides storage for a fixed
//! number of one type, while [`UntypedBlock`] provides a fixed number of bytes
//! that may be reused for different layouts.
//!
//! Both stack types preserve insertion order when viewed as a slice and remove
//! values in last-in, first-out order. Their capacity never grows.
//!
//! ## Installation
//!
//! This crate is experimental, and its public API may change between `0.x`
//! releases. Prefer the broad `"0"` version requirement so Cargo can select
//! newer pre-1.0 releases, including releases with potentially API-breaking
//! changes.
//!
//! The `alloc` feature is enabled by default. To omit the heap-backed `Box` and
//! `Vec` convenience APIs, disable default features in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! stackyard = { version = "0", default-features = false }
//! ```
//!
//! ## Quick Start
//!
//! ```rust
//! use stackyard::InlineStack;
//!
//! let mut stack = InlineStack::<u8, 2>::new();
//! stack.push(1);
//! stack.push(2);
//! assert_eq!(stack.pop(), Some(2));
//! ```
//!
//! [`InlineStack::push`] deliberately returns `()` and silently drops its input
//! when the stack is full. Use [`InlineStack::try_push`] when the rejected value
//! must be recovered.
//!
//! ## Borrowed Storage
//!
//! Use [`Stack`] when capacity is known only at run time or when the stack
//! handle should remain small. The allocator owns the storage and the stack
//! borrows it until the stack is dropped.
//!
//! ```rust
//! use stackyard::{Stack, TypedBlock};
//!
//! let block = TypedBlock::<u8, 4>::new();
//! let mut stack = Stack::<u8, _>::try_new_in(2, &block).unwrap();
//! stack.push(1);
//! assert_eq!(stack.as_slice(), &[1]);
//! ```
//!
//! [`TypedBlock`] is the direct choice when the element type is known.
//! [`UntypedBlock`] can serve different layouts sequentially, but alignment
//! padding can reduce its usable capacity. The low-level [`alloc`] module
//! exposes the [`Alloc`] raw-storage contract; collections remain responsible
//! for initializing and dropping values placed in a lease.
//!
//! ## Feature Flags
//!
//! - `alloc` *(default)* — enables the `Box` and `Vec` convenience APIs. All
//!   fixed-capacity stacks and block allocators remain available without it.
//!
//! ## License
//!
//! Licensed under the GNU Affero General Public License, version 3 only
//! (`AGPL-3.0-only`).
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc as rust_alloc;

pub mod alloc;
mod stack;

pub use alloc::{Alloc, TypedBlock, UntypedBlock};
pub use stack::{InlineStack, Stack};
