// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
//! # stackyard
//!
//! Stack-based data structures and algorithms for `no_std` programs.
//!
//! The crate focuses on predictable storage requirements and supports optional
//! allocation-dependent conveniences.
//!
//! ## Installation
//!
//! This crate is experimental, and its public API may change between `0.x`
//! releases. Prefer the broad `"0"` version requirement so Cargo can select
//! newer pre-1.0 releases, including releases with potentially API-breaking
//! changes.
//!
//! The `alloc` feature is enabled by default. To use only allocation-free APIs,
//! disable default features in your `Cargo.toml`:
//!
//! ```yaml
//! [dependencies]
//! stackyard = { version = "0", default-features = false }
//! ```
//!
//! ## Quick Start
//!
//! ```rust
//! use stackyard::Stack;
//!
//! let mut stack = Stack::<u8, 3>::new();
//! assert_eq!(stack.push(10), None);
//! assert_eq!(stack.push(20), None);
//! assert_eq!(stack.as_slice(), &[10, 20]);
//! assert_eq!(stack.pop(), Some(20));
//! ```
//!
//! ## Feature Flags
//!
//! - `alloc` *(default)* — enables APIs that require an allocator. Core
//!   fixed-capacity APIs remain available without it.
//!
//! ## License
//!
//! Licensed under the GNU Affero General Public License, version 3 only
//! (`AGPL-3.0-only`).
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

mod stack;

pub use stack::Stack;
