//! The caches of a layer stack: their detach and their cache traversal.

use super::*;
use crate::utils::test_helpers::panic_text;

/// `cache_detach` on the caches of a stack puts each slot back onto the
/// device of the stack. A plain slot stays plain, and an autodiff slot keeps
/// the checkpointing strategy of that device.
#[test]
fn detach_keeps_the_device_of_each_slot() {
    for device in [test_device(), test_device().autodiff().gradient_checkpointing()] {
        let (_, caches) = layers(2, &device).forward(randn3(1, 3, &device), None, (), None, None);
        for slot in cache_detach(caches) {
            let state = slot.state_bd;
            assert_eq!(state.is_autodiff(), device.is_autodiff());
            assert_eq!(
                state.gradient_checkpointing_strategy(),
                device.gradient_checkpointing_strategy(),
            );
        }
    }
}

/// The cache traversal of the caches of a stack moves each slot to its own
/// place. A stack of 2 layers gives two caches: the first after one window,
/// the second after one more window. `assign_in_place` of the second caches
/// into the owned buffers of the first caches gives each slot of the second
/// caches, in its place. A traversal of two lists with different lengths
/// panics.
#[test]
fn the_caches_of_a_stack_move_each_slot() {
    let device = test_device();
    let layers = layers(2, &device);
    let (_, first) = layers.forward(randn3(2, 3, &device), None, (), None, None);
    let (_, second) = layers.forward(randn3(2, 3, &device), Some(first.clone()), (), None, None);
    assert_eq!(second.len(), 2, "one cache for each layer");

    let moved = first.clone().into_owned_buffers().assign_in_place(second.clone());
    assert_eq!(moved.len(), second.len(), "the moved caches keep the slot count");
    for (i, (got, want)) in moved.into_iter().zip(&second).enumerate() {
        let diff = max_rel_diff(got.state_bd, want.state_bd.clone());
        assert!(diff < dtype_tol(TOL), "slot {i} of the moved caches and of their source differ by {diff}");
    }

    let shorter = second[..1].to_vec();
    let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| first.assign_in_place(shorter)))
    else {
        panic!("a traversal of two lists with different lengths does not panic");
    };
    let message = panic_text(&*payload).unwrap_or_default();
    assert!(
        message.contains("the two caches differ in slot count"),
        "the panic `{message}` does not name the slot count",
    );
}
