//! The tests of the backend hop that follows from the cache traversal.

use super::*;

/// The device of a cache comes from its tensors. So a cache with no tensor
/// has no device, and `cache_device` refuses it.
#[test]
#[should_panic(expected = "the cache holds no tensor")]
fn cache_device_refuses_a_cache_with_no_tensor() {
    let _ = cache_device(&());
}
