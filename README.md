# stackyard

Freestanding data structures and algorithms.

## Installation

This crate is experimental, and its public API may change between `0.x`
releases. Prefer the broad `"0"` version requirement so Cargo can select newer
pre-1.0 releases, including releases with potentially API-breaking changes.

The `alloc` feature is enabled by default. To use only allocation-free APIs,
disable default features in your `Cargo.toml`:

```toml
[dependencies]
stackyard = { version = "0", default-features = false }
```

## Quick Start

```rust
use stackyard::InlineStack;

let mut stack = InlineStack::<u8, 2>::new();
stack.push(10);
stack.push(20);
stack.push(30); // Full: 30 is silently dropped.
assert_eq!(stack.as_slice(), &[10, 20]);
assert_eq!(stack.last(), Some(&20));
assert_eq!(stack.try_push(30), Some(30));
assert_eq!(stack.pop(), Some(20));
```

`InlineStack::push` deliberately returns `()` and silently drops its input when
the stack is full. This keeps it at performance parity with `Vec::push` after
the `Vec` has allocated sufficient capacity. Use `InlineStack::try_push` when
the rejected value must be recovered.

## Feature Flags

- `alloc` *(default)* — enables APIs that require an allocator. Core
  fixed-capacity APIs remain available without it.

## License

Licensed under the GNU Affero General Public License, version 3 only
(`AGPL-3.0-only`).
