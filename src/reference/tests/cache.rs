//! The cache stack: its detach.

use super::*;

/// `CacheStack::detach` puts each slot back on its own device: a plain slot
/// stays plain, and an autodiff slot keeps the checkpointing strategy of its
/// device.
#[test]
fn detach_keeps_the_device_of_each_slot() {
    for device in [test_device(), test_device().autodiff().gradient_checkpointing()] {
        let (_, caches) = layers(2, &device).forward(randn3(1, 3, &device), None, (), None, None);
        for slot in caches.detach().into_slots().into_iter().flatten() {
            let state = slot.state_bd;
            assert_eq!(state.is_autodiff(), device.is_autodiff());
            assert_eq!(
                state.gradient_checkpointing_strategy(),
                device.gradient_checkpointing_strategy(),
            );
        }
    }
}
