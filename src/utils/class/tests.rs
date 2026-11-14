//! The placement plans of the class markers, with no container: the output
//! indices, the window plan and the prime plan.

use super::*;

/// A `Custom(k)` lands before the user token `k`. Each marker before it
/// moves it one output position further.
#[test]
fn class_marker_output_indices_shift_a_custom_index() {
    let markers = vec![ClassLatent::Custom(1), ClassLatent::Custom(3)];
    // L = 5 ⇒ u0 C u1 u2 C u3 u4. The second `Custom` precedes u3 at the
    // output position 4, because the first one lands before it.
    assert_eq!(class_marker_output_indices(&markers, 5), vec![1, 4]);
}

/// The window plan in four cases:
///
/// 1. the four marker kinds in one window for the whole sequence,
/// 2. the same sequence split in two windows,
/// 3. a `Custom` at the announced end, and a token past that end,
/// 4. `Start` and `Custom` in an open-ended stream.
#[test]
fn class_window_plan_splits_a_sequence() {
    let markers = vec![
        ClassLatent::Start,
        ClassLatent::Middle,
        ClassLatent::End,
        ClassLatent::Custom(1),
    ];
    // L = 6 ⇒ S u0 C u1 u2 M u3 u4 u5 E. In `Vec` order (Start, Middle, End,
    // Custom), the positions are [0, 5, 9, 2].
    assert_eq!(class_marker_output_indices(&markers, 6), vec![0, 5, 9, 2]);

    // One window for the whole sequence. Each pair is `(at, marker)`, and `at`
    // counts the tokens of the window before the marker.
    let mut cursor = ClassCursor::whole(6);
    assert_eq!(
        class_window_plan(&markers, 6, &mut cursor, "test"),
        vec![(0, 0), (1, 3), (3, 1), (6, 2)]
    );
    assert_eq!(cursor.offset, 10);

    // Two windows of 4 and 2 tokens give the same placement. `End` waits for
    // the window that closes the sequence.
    let mut cursor = ClassCursor::whole(6);
    assert_eq!(
        class_window_plan(&markers, 4, &mut cursor, "test"),
        vec![(0, 0), (1, 3), (3, 1)]
    );
    assert_eq!(cursor.offset, 7);
    assert_eq!(class_window_plan(&markers, 2, &mut cursor, "test"), vec![(2, 2)]);
    assert_eq!(cursor.offset, 10);

    // `End` closes the sequence, and `Custom` does not. A `Custom` at the end
    // has no token to precede, so it waits for a further token.
    let markers = vec![ClassLatent::End, ClassLatent::Custom(3)];
    // L = 3 ⇒ u0 u1 u2 E. The `Custom` does not land at its position 4.
    assert_eq!(class_marker_output_indices(&markers, 3), vec![3, 4]);
    let mut cursor = ClassCursor::whole(3);
    assert_eq!(class_window_plan(&markers, 3, &mut cursor, "test"), vec![(3, 0)]);
    assert_eq!(cursor.offset, 4);
    // One token after the announced length, the `Custom` lands. It precedes
    // that token, as each `Custom` does.
    assert_eq!(class_window_plan(&markers, 1, &mut cursor, "test"), vec![(0, 1)]);
    assert_eq!(cursor.offset, 6);

    // An open-ended stream has no length hint. `Start` and `Custom` still
    // land at their places, and no marker closes the stream.
    let markers = vec![ClassLatent::Start, ClassLatent::Custom(1)];
    let mut cursor = ClassCursor::default();
    assert_eq!(class_window_plan(&markers, 1, &mut cursor, "test"), vec![(0, 0)]);
    assert_eq!(cursor.offset, 2);
    assert_eq!(class_window_plan(&markers, 1, &mut cursor, "test"), vec![(0, 1)]);
    assert_eq!(cursor.offset, 4);
}

/// The prime plan against the window plan. A prime emits the markers that
/// wait for the next user token, at the cursor. It leaves each other marker
/// for the step that finds it. A prime never emits `End`. A cursor at the
/// announced end primes nothing.
#[test]
fn class_prime_plan_emits_only_what_waits_for_the_next_token() {
    let markers = vec![ClassLatent::Start, ClassLatent::Custom(2), ClassLatent::End];
    // L = 3 ⇒ S u0 u1 C u2 E, at the positions [0, 3, 5].
    assert_eq!(class_marker_output_indices(&markers, 3), vec![0, 3, 5]);

    // A prime at the start emits `Start` alone. The other markers wait for a
    // token.
    let mut cursor = ClassCursor::whole(3);
    assert_eq!(class_prime_plan(&markers, &mut cursor, "test"), vec![(0, 0)]);
    assert_eq!(cursor.offset, 1);
    // A second prime finds nothing.
    assert_eq!(class_prime_plan(&markers, &mut cursor, "test"), vec![]);
    assert_eq!(cursor.offset, 1);
    // The next two steps carry only their user tokens. `Start` is behind the
    // cursor, and the `Custom` waits for u2.
    assert_eq!(class_window_plan(&markers, 1, &mut cursor, "test"), vec![]);
    assert_eq!(class_window_plan(&markers, 1, &mut cursor, "test"), vec![]);
    assert_eq!(cursor.offset, 3);
    // The `Custom` precedes u2, so a prime emits it now.
    assert_eq!(class_prime_plan(&markers, &mut cursor, "test"), vec![(0, 1)]);
    assert_eq!(cursor.offset, 4);
    // The last user token closes the sequence, and `End` follows it in that
    // step. The prime after it has no next token, so it emits nothing.
    assert_eq!(class_window_plan(&markers, 1, &mut cursor, "test"), vec![(1, 2)]);
    assert_eq!(cursor.offset, 6);
    assert_eq!(class_prime_plan(&markers, &mut cursor, "test"), vec![]);
    assert_eq!(cursor.offset, 6);

    // A `Custom` at the announced end waits for a token past that end. A
    // prime at the end has no next token, so it does not emit the `Custom`.
    let markers = vec![ClassLatent::End, ClassLatent::Custom(3)];
    let mut cursor = ClassCursor::whole(3);
    assert_eq!(class_window_plan(&markers, 3, &mut cursor, "test"), vec![(3, 0)]);
    assert_eq!(class_prime_plan(&markers, &mut cursor, "test"), vec![]);
    assert_eq!(cursor.offset, 4);

    // An open-ended stream has no length hint, so a further token always
    // comes. A prime emits each `Start` and `Custom` at its place.
    let markers = vec![ClassLatent::Start, ClassLatent::Custom(1)];
    let mut cursor = ClassCursor::default();
    assert_eq!(class_prime_plan(&markers, &mut cursor, "test"), vec![(0, 0)]);
    assert_eq!(cursor.offset, 1);
    assert_eq!(class_window_plan(&markers, 1, &mut cursor, "test"), vec![]);
    assert_eq!(cursor.offset, 2);
    assert_eq!(class_prime_plan(&markers, &mut cursor, "test"), vec![(0, 1)]);
    assert_eq!(cursor.offset, 3);
}
