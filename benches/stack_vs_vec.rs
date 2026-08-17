// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

//! Raw operation throughput for stackyard collections against a preallocated
//! `Vec`.
//!
//! Destination construction, `Vec` allocation, and source population happen
//! in setup outside the measured routine. Multi-operation samples use
//! `ELEMENTS` as their throughput so the collection implementations remain
//! directly comparable.
//!
//! The slice groups include both the literal `Vec::extend` spelling and
//! `Vec::extend_from_slice`, the closest specialized standard-library API.

use std::{array, hint::black_box, ptr::NonNull, rc::Rc};

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use stackyard::{InlineStack, Stack, TypedBlock};

const ELEMENTS: usize = 256;

type BenchBlock<T> = TypedBlock<T, ELEMENTS>;
type BenchStack<T> = Stack<'static, T, BenchBlock<T>>;

/// A movable benchmark input whose backing block remains at a stable address.
///
/// `stack` is always dropped before `_block`, so its extended allocator
/// lifetime never escapes this wrapper.
struct StackInput<T: 'static> {
    stack: Option<BenchStack<T>>,
    _block: Box<BenchBlock<T>>,
}

impl<T: 'static> StackInput<T> {
    fn new() -> Self {
        let block = Box::new(BenchBlock::new());
        let block_ptr = NonNull::from(block.as_ref());

        // SAFETY:
        // - `block` owns a heap allocation whose address remains stable when
        //   the Box or enclosing StackInput moves
        // - StackInput never exposes the extended reference or owned Stack
        // - StackInput::drop destroys `stack` before releasing `_block`
        let allocator: &'static BenchBlock<T> = unsafe { block_ptr.as_ref() };
        let stack = Stack::try_new_in(ELEMENTS, allocator)
            .expect("a fresh exact-capacity TypedBlock accepts one Stack");

        Self {
            stack: Some(stack),
            _block: block,
        }
    }

    #[inline]
    fn stack_mut(&mut self) -> &mut BenchStack<T> {
        self.stack.as_mut().expect("the Stack is live")
    }
}

impl<T: 'static> Drop for StackInput<T> {
    fn drop(&mut self) {
        drop(self.stack.take());
    }
}

fn values() -> [u64; ELEMENTS] {
    array::from_fn(|index| (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

fn push(c: &mut Criterion) {
    let values = values();
    let mut group = c.benchmark_group("push_after_allocation");
    group.throughput(Throughput::Elements(ELEMENTS as u64));

    group.bench_function("InlineStack::try_push", |b| {
        b.iter_batched_ref(
            InlineStack::<u64, ELEMENTS>::new,
            |stack| {
                let stack = black_box(stack);
                for &value in black_box(values.as_slice()) {
                    let rejected = stack.try_push(value);
                    debug_assert!(rejected.is_none());
                }
                black_box(stack.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("InlineStack::push", |b| {
        b.iter_batched_ref(
            InlineStack::<u64, ELEMENTS>::new,
            |stack| {
                let stack = black_box(stack);
                for &value in black_box(values.as_slice()) {
                    stack.push(value);
                }
                black_box(stack.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Stack<TypedBlock>::push", |b| {
        b.iter_batched_ref(
            StackInput::<u64>::new,
            |input| {
                let stack = black_box(input.stack_mut());
                for &value in black_box(values.as_slice()) {
                    stack.push(value);
                }
                black_box(stack.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Vec::push", |b| {
        b.iter_batched_ref(
            || Vec::<u64>::with_capacity(ELEMENTS),
            |values_out| {
                let values_out = black_box(values_out);
                for &value in black_box(values.as_slice()) {
                    values_out.push(value);
                }
                black_box(values_out.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn pop(c: &mut Criterion) {
    let values = values();
    let mut group = c.benchmark_group("pop");
    group.throughput(Throughput::Elements(ELEMENTS as u64));

    group.bench_function("InlineStack::pop", |b| {
        b.iter_batched_ref(
            || {
                let mut stack = InlineStack::<u64, ELEMENTS>::new();
                assert!(stack.copy_from_slice(&values).is_none());
                stack
            },
            |stack| {
                let stack = black_box(stack);
                let mut checksum = 0u64;
                for _ in 0..ELEMENTS {
                    checksum = checksum.wrapping_add(stack.pop().expect("stack was populated"));
                }
                debug_assert!(stack.is_empty());
                black_box(checksum);
                black_box(stack.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Stack<TypedBlock>::pop", |b| {
        b.iter_batched_ref(
            || {
                let mut input = StackInput::<u64>::new();
                assert!(input.stack_mut().copy_from_slice(&values).is_none());
                input
            },
            |input| {
                let stack = black_box(input.stack_mut());
                let mut checksum = 0u64;
                for _ in 0..ELEMENTS {
                    checksum = checksum.wrapping_add(stack.pop().expect("stack was populated"));
                }
                debug_assert!(stack.is_empty());
                black_box(checksum);
                black_box(stack.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Vec::pop", |b| {
        b.iter_batched_ref(
            || values.to_vec(),
            |values_out| {
                let values_out = black_box(values_out);
                let mut checksum = 0u64;
                for _ in 0..ELEMENTS {
                    checksum = checksum.wrapping_add(values_out.pop().expect("vec was populated"));
                }
                debug_assert!(values_out.is_empty());
                black_box(checksum);
                black_box(values_out.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn copy_from_slice(c: &mut Criterion) {
    let values = values();
    let mut group = c.benchmark_group("copy_from_slice");
    group.throughput(Throughput::Elements(ELEMENTS as u64));

    group.bench_function("InlineStack::copy_from_slice", |b| {
        b.iter_batched_ref(
            InlineStack::<u64, ELEMENTS>::new,
            |stack| {
                let stack = black_box(stack);
                let remainder = stack.copy_from_slice(black_box(values.as_slice()));
                debug_assert!(remainder.is_none());
                black_box(stack.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Stack<TypedBlock>::copy_from_slice", |b| {
        b.iter_batched_ref(
            StackInput::<u64>::new,
            |input| {
                let stack = black_box(input.stack_mut());
                let remainder = stack.copy_from_slice(black_box(values.as_slice()));
                debug_assert!(remainder.is_none());
                black_box(stack.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Vec::extend", |b| {
        b.iter_batched_ref(
            || Vec::<u64>::with_capacity(ELEMENTS),
            |values_out| {
                let values_out = black_box(values_out);
                values_out.extend(black_box(values.as_slice()).iter().copied());
                black_box(values_out.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Vec::extend_from_slice", |b| {
        b.iter_batched_ref(
            || Vec::<u64>::with_capacity(ELEMENTS),
            |values_out| {
                let values_out = black_box(values_out);
                values_out.extend_from_slice(black_box(values.as_slice()));
                black_box(values_out.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn clone_from_slice(c: &mut Criterion) {
    // `Rc` makes this a real, cheap Clone workload rather than allowing the
    // optimizer to turn a clone-only wrapper into the same memcpy as `Copy`.
    let values: [Rc<u64>; ELEMENTS] = array::from_fn(|index| Rc::new(index as u64));
    let mut group = c.benchmark_group("clone_from_slice");
    group.throughput(Throughput::Elements(ELEMENTS as u64));

    group.bench_function("InlineStack::clone_from_slice", |b| {
        b.iter_batched_ref(
            InlineStack::<Rc<u64>, ELEMENTS>::new,
            |stack| {
                let stack = black_box(stack);
                let remainder = stack.clone_from_slice(black_box(values.as_slice()));
                debug_assert!(remainder.is_none());
                black_box(stack.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Stack<TypedBlock>::clone_from_slice", |b| {
        b.iter_batched_ref(
            StackInput::<Rc<u64>>::new,
            |input| {
                let stack = black_box(input.stack_mut());
                let remainder = stack.clone_from_slice(black_box(values.as_slice()));
                debug_assert!(remainder.is_none());
                black_box(stack.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Vec::extend", |b| {
        b.iter_batched_ref(
            || Vec::<Rc<u64>>::with_capacity(ELEMENTS),
            |values_out| {
                let values_out = black_box(values_out);
                values_out.extend(black_box(values.as_slice()).iter().cloned());
                black_box(values_out.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Vec::extend_from_slice", |b| {
        b.iter_batched_ref(
            || Vec::<Rc<u64>>::with_capacity(ELEMENTS),
            |values_out| {
                let values_out = black_box(values_out);
                values_out.extend_from_slice(black_box(values.as_slice()));
                black_box(values_out.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, push, pop, copy_from_slice, clone_from_slice,);
criterion_main!(benches);
