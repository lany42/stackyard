// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::Rc,
    string::String,
};

use stackyard::{Stack, UntypedBlock};

#[test]
fn constructor_failures_leave_the_block_available_for_an_exact_layout() {
    struct Zst;

    let block = UntypedBlock::<16>::new();

    assert!(Stack::<u8, _>::try_new_in(0, &block).is_none());
    assert!(Stack::<Zst, _>::try_new_in(1, &block).is_none());
    assert!(Stack::<u16, _>::try_new_in(usize::MAX, &block).is_none());
    assert!(Stack::<u8, _>::try_new_in(17, &block).is_none());

    let mut stack = Stack::<u8, _>::try_new_in(16, &block).expect("the exact byte layout fits");
    assert_eq!(stack.capacity(), 16);
    assert_eq!(stack.copy_from_slice(&[7; 16]), None);
    assert_eq!(stack.as_slice(), &[7; 16]);

    assert!(Stack::<u8, _>::try_new_in(1, &block).is_none());
    assert_eq!(stack.as_slice(), &[7; 16]);

    drop(stack);

    let reused = Stack::<u8, _>::try_new_in(16, &block)
        .expect("dropping the exact-fit Stack releases the block");
    drop(reused);
}

struct Tracked {
    value: u8,
    drops: Rc<Cell<usize>>,
}

impl Tracked {
    fn new(value: u8, drops: &Rc<Cell<usize>>) -> Self {
        Self {
            value,
            drops: Rc::clone(drops),
        }
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

#[test]
fn push_pop_clear_and_drop_transfer_each_value_exactly_once() {
    let drops = Rc::new(Cell::new(0));
    let block = UntypedBlock::<256>::new();
    let mut stack = Stack::try_new_in(3, &block).expect("three tracked values fit");

    for value in [10, 20, 30] {
        assert!(stack.try_push(Tracked::new(value, &drops)).is_none());
    }
    assert_eq!(stack.len(), 3);
    assert!(stack.is_full());
    assert_eq!(
        stack
            .as_slice()
            .iter()
            .map(|tracked| tracked.value)
            .collect::<Vec<_>>(),
        [10, 20, 30]
    );

    let rejected = stack
        .try_push(Tracked::new(40, &drops))
        .expect("a full Stack returns the rejected value");
    assert_eq!(rejected.value, 40);
    drop(rejected);
    assert_eq!(drops.get(), 1);

    let popped = stack.pop().expect("the Stack is not empty");
    assert_eq!(popped.value, 30);
    drop(popped);
    assert_eq!(drops.get(), 2);

    stack.as_mut_slice()[0].value = 11;
    assert_eq!(stack.last().map(|tracked| tracked.value), Some(20));
    assert_eq!(
        stack
            .as_slice()
            .iter()
            .map(|tracked| tracked.value)
            .collect::<Vec<_>>(),
        [11, 20]
    );

    stack.clear();
    assert!(stack.is_empty());
    assert!(stack.as_slice().is_empty());
    assert_eq!(drops.get(), 4);

    for value in [50, 60, 70] {
        stack.push(Tracked::new(value, &drops));
    }
    stack.push(Tracked::new(80, &drops));
    assert_eq!(drops.get(), 5);

    drop(stack);
    assert_eq!(drops.get(), 8);

    let reused = Stack::<u8, _>::try_new_in(1, &block)
        .expect("normal Stack destruction releases the untyped block");
    drop(reused);
}

#[test]
fn copy_clone_and_fill_work_across_sequentially_retyped_leases() {
    let block = UntypedBlock::<255>::new();

    {
        let mut stack = Stack::<u16, _>::try_new_in(4, &block).expect("four u16 values fit");
        let input = [10, 20, 30, 40, 50];

        assert_eq!(stack.copy_from_slice(&input), Some(&input[4..]));
        assert_eq!(stack.as_slice(), [10, 20, 30, 40]);
        for value in &mut stack {
            *value += 1;
        }
        assert_eq!(stack.as_slice(), [11, 21, 31, 41]);
    }

    {
        let mut stack =
            Stack::<String, _>::try_new_in(3, &block).expect("three String values fit after reuse");
        stack.push(String::from("existing"));
        let input = [
            String::from("left"),
            String::from("right"),
            String::from("remainder"),
        ];

        assert_eq!(stack.clone_from_slice(&input), Some(&input[2..]));
        assert_eq!(stack.as_slice(), ["existing", "left", "right"]);

        stack.clear();
        stack.push(String::from("seed"));
        stack.fill(String::from("fill"));
        assert_eq!(stack.as_slice(), ["seed", "fill", "fill"]);
    }

    let reused = Stack::<u8, _>::try_new_in(255, &block)
        .expect("sequential leases may use unrelated element types");
    drop(reused);
}

#[repr(align(64))]
#[derive(Debug, Eq, PartialEq)]
struct OverAligned(u8);

#[test]
fn over_aligned_elements_use_a_fitting_interior_pointer() {
    let block = UntypedBlock::<127>::new();
    let mut stack =
        Stack::<OverAligned, _>::try_new_in(1, &block).expect("worst-case alignment padding fits");

    stack.push(OverAligned(7));

    assert_eq!(stack.as_slice().as_ptr().addr() % 64, 0);
    assert_eq!(stack.last(), Some(&OverAligned(7)));
    assert_eq!(stack.pop(), Some(OverAligned(7)));
}

#[repr(align(2))]
struct AlignTwo(u8);

#[test]
fn alignment_padding_can_reject_an_exact_sized_backing_array() {
    let blocks = [UntypedBlock::<2>::new(), UntypedBlock::<2>::new()];
    let first = Stack::<AlignTwo, _>::try_new_in(1, &blocks[0]);
    let second = Stack::<AlignTwo, _>::try_new_in(1, &blocks[1]);
    let first_succeeded = first.is_some();

    assert_eq!(
        usize::from(first_succeeded) + usize::from(second.is_some()),
        1
    );

    let failed_index = usize::from(first_succeeded);
    for mut stack in [first, second].into_iter().flatten() {
        stack.push(AlignTwo(9));
        assert_eq!(stack.as_slice().as_ptr().addr() % 2, 0);
        assert_eq!(stack.last().map(|value| value.0), Some(9));
    }

    let mut bytes = Stack::<u8, _>::try_new_in(2, &blocks[failed_index])
        .expect("padding failure leaves the untyped block available");
    assert_eq!(bytes.copy_from_slice(&[1, 2]), None);
    assert_eq!(bytes.as_slice(), [1, 2]);
}

#[test]
fn a_live_lease_excludes_another_stack_without_touching_its_bytes() {
    let block = UntypedBlock::<128>::new();
    let mut first = Stack::<u64, _>::try_new_in(4, &block).expect("four u64 values fit");
    assert_eq!(first.copy_from_slice(&[10, 20, 30, 40]), None);

    {
        let leased = first.as_mut_slice();

        assert!(Stack::<u8, _>::try_new_in(1, &block).is_none());
        assert_eq!(leased, [10, 20, 30, 40]);
        leased[0] = 11;
    }

    assert_eq!(first.as_slice(), [11, 20, 30, 40]);
    drop(first);

    let mut second = Stack::<u8, _>::try_new_in(128, &block)
        .expect("releasing the first lease permits a differently typed lease");
    assert_eq!(second.copy_from_slice(&[1, 2, 3]), None);
    assert_eq!(second.as_slice(), [1, 2, 3]);
}

struct CloneTracked {
    value: u8,
    clones: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
}

impl Clone for CloneTracked {
    fn clone(&self) -> Self {
        self.clones.set(self.clones.get() + 1);
        Self {
            value: self.value,
            clones: Rc::clone(&self.clones),
            drops: Rc::clone(&self.drops),
        }
    }
}

impl Drop for CloneTracked {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

#[test]
fn clone_allocation_failure_is_transactional_and_success_is_independent() {
    let clones = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let source_block = UntypedBlock::<256>::new();
    let small_block = UntypedBlock::<1>::new();
    let destination_block = UntypedBlock::<256>::new();
    let mut source = Stack::try_new_in(3, &source_block).expect("three clone-tracked values fit");

    for value in [10, 20] {
        source.push(CloneTracked {
            value,
            clones: Rc::clone(&clones),
            drops: Rc::clone(&drops),
        });
    }

    assert!(source.try_clone_into(&source_block).is_none());
    assert!(source.try_clone_into(&small_block).is_none());
    assert_eq!(clones.get(), 0);
    assert_eq!(drops.get(), 0);
    assert_eq!(
        source
            .as_slice()
            .iter()
            .map(|value| value.value)
            .collect::<Vec<_>>(),
        [10, 20]
    );

    let small_reused = Stack::<u8, _>::try_new_in(1, &small_block)
        .expect("failed cloning does not occupy the destination allocator");
    drop(small_reused);

    let cloned = source
        .try_clone_into(&destination_block)
        .expect("the destination grants a disjoint lease");
    assert_eq!(clones.get(), 2);
    source.as_mut_slice()[0].value = 11;
    assert_eq!(source.as_slice()[0].value, 11);
    assert_eq!(cloned.as_slice()[0].value, 10);

    drop(cloned);
    assert_eq!(drops.get(), 2);
    drop(source);
    assert_eq!(drops.get(), 4);
}

struct PanicOnSecondClone {
    clone_calls: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
}

impl Clone for PanicOnSecondClone {
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

impl Drop for PanicOnSecondClone {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

#[test]
fn clone_unwinding_drops_partial_clones_and_releases_the_destination() {
    let clone_calls = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let source_block = UntypedBlock::<256>::new();
    let destination_block = UntypedBlock::<256>::new();
    let mut source = Stack::try_new_in(3, &source_block).expect("three clone-tracked values fit");

    for _ in 0..2 {
        source.push(PanicOnSecondClone {
            clone_calls: Rc::clone(&clone_calls),
            drops: Rc::clone(&drops),
        });
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        source.try_clone_into(&destination_block)
    }));

    assert!(result.is_err());
    assert_eq!(clone_calls.get(), 2);
    assert_eq!(drops.get(), 1);
    assert_eq!(source.len(), 2);

    let reused = Stack::<u8, _>::try_new_in(1, &destination_block)
        .expect("the partial destination releases its lease while unwinding");
    drop(reused);

    drop(source);
    assert_eq!(drops.get(), 3);
}

struct PanicsOnceOnDrop {
    panicked: Rc<Cell<bool>>,
    drops: Rc<Cell<usize>>,
}

impl Drop for PanicsOnceOnDrop {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
        if !self.panicked.replace(true) {
            panic!("requested destructor panic");
        }
    }
}

#[test]
fn destructor_unwinding_releases_the_untyped_block() {
    let panicked = Rc::new(Cell::new(false));
    let drops = Rc::new(Cell::new(0));
    let block = UntypedBlock::<128>::new();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut stack = Stack::try_new_in(1, &block).expect("one panicking value fits");
        stack.push(PanicsOnceOnDrop {
            panicked: Rc::clone(&panicked),
            drops: Rc::clone(&drops),
        });
    }));

    assert!(result.is_err());
    assert!(panicked.get());
    assert_eq!(drops.get(), 1);

    let reused = Stack::<u8, _>::try_new_in(128, &block)
        .expect("the release guard frees the untyped block during unwinding");
    drop(reused);
}
