// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
//! Stack data structures.

mod inline_stack;

pub use inline_stack::InlineStack;

/// Placeholder for the allocator-backed stack implementation.
pub struct Stack {}
