// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

mod stack;

pub use stack::Stack;
