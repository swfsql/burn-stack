//! # Recompute node: one autodiff node for a custom backward with two outputs
//!
//! A recompute backward saves the inputs of an op, and it does not save the
//! intermediates of the forward. The backward computes the gradients from
//! these inputs. An op that returns a sequence output and a final state has
//! two outputs, and one [`Backward`](burn::backend::autodiff::ops::Backward)
//! node serves both.
//!
//! A caller gives named inputs, one `Extra` value and one
//! [`RecomputeOp`](crate::utils::recompute_node::RecomputeOp) type.
//! [`recompute_pair`](crate::utils::recompute_node::recompute_pair) owns the
//! autodiff wiring:
//!
//! - It calls `prepare` with the parents in the order of the named inputs.
//! - It runs the tracked arm or the untracked arm.
//! - It packs the two outputs, and it splits them at the output shapes.
//! - It consumes the gradient of the node, and it registers one gradient for
//!   each parent.
//!
//! It reads the output shapes from the results of the forward, so a caller
//! gives no shape and no length.
//!
//! ## One tracked tensor
//!
//! `OpsPrep::finish` takes one output tensor, and a node registers one graph
//! step. So the node packs the two outputs into one tracked 1-D tensor, and it
//! returns two views at the output shapes. Autodiff adds the gradients of the
//! two views into the gradient of the tensor. The backward splits that
//! gradient again.
//!
//! The untracked arm has no gradient to serve, but it packs and splits the
//! outputs in the same way. `OpsPrep<UnTracked>::finish` also returns one
//! tensor, and it registers a graph step on purpose. A second output without
//! the `float_cat` needs a second prepare or a bare leaf.
//!
//! ## Named inputs
//!
//! `prepare`, `Ops::parents` and the saved state each hold the inputs as a
//! positional array. A wrong order compiles, and it gives wrong gradients. So
//! an op declares its inputs as one struct with
//! [`recompute_inputs!`](crate::recompute_inputs), and the macro holds the
//! only conversion to and from the array. The backward of
//! an op returns the same struct, with one optional gradient for each field.
//! So the compiler refuses an op that forgets a gradient.

use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, TensorMetadata};
use burn::prelude::*;
use core::fmt;
use core::marker::PhantomData;

/// The float primitive of `B`.
pub type Prim<B> = FloatTensor<B>;

/// The named inputs of a [`RecomputeOp`], with one value of `T` for each
/// input. [`recompute_inputs!`](crate::recompute_inputs) implements this
/// trait.
pub trait RecomputeInputs<T, const N: usize>: Sized {
    /// The values, in the order of the fields.
    fn into_array(self) -> [T; N];
    /// The inverse of [`into_array`](Self::into_array).
    fn from_array(array: [T; N]) -> Self;
}

/// Declare `struct Name<T>` with one field of type `T` for each name, and its
/// [`RecomputeInputs`] impl. The macro counts `N` from the fields.
///
/// `recompute_inputs!(Name { field, .. })` declares a private struct. The
/// [`RecomputeOp`] impl and the call of [`recompute_pair`] are in the module
/// of the macro call.
///
/// [`RecomputeInputs`]: crate::utils::recompute_node::RecomputeInputs
/// [`RecomputeOp`]: crate::utils::recompute_node::RecomputeOp
/// [`recompute_pair`]: crate::utils::recompute_node::recompute_pair
#[macro_export]
macro_rules! recompute_inputs {
    ($name:ident { $($field:ident),+ $(,)? }) => {
        struct $name<T> { $($field: T),+ }

        impl<T> $crate::utils::recompute_node::RecomputeInputs<
            T,
            { 0 $(+ $crate::recompute_inputs!(@one $field))+ },
        > for $name<T> {
            fn into_array(self) -> [T; { 0 $(+ $crate::recompute_inputs!(@one $field))+ }] {
                [$(self.$field),+]
            }

            fn from_array(array: [T; { 0 $(+ $crate::recompute_inputs!(@one $field))+ }]) -> Self {
                let [$($field),+] = array;
                Self { $($field),+ }
            }
        }
    };
    (@one $field:ident) => { 1usize };
}

/// One op with two outputs and a recompute backward: its inputs, its constants,
/// its forward on the inner backend and its gradient math.
pub trait RecomputeOp<B: Backend, const N: usize>: 'static {
    /// The named inputs. Each one is a parent of the node.
    type Inputs<T>: RecomputeInputs<T, N>;
    /// The values that the op takes with no gradient.
    type Extra: Clone + fmt::Debug + Send + 'static;

    /// The forward on the inner backend. Returns `y` and `final_state`.
    fn forward(inputs: Self::Inputs<Prim<B>>, extra: &Self::Extra) -> (Prim<B>, Prim<B>);

    /// The gradient of each input, from the gradients of the two outputs. A
    /// `None` gradient registers nothing.
    fn backward(
        d_y: Prim<B>,
        d_final_state: Prim<B>,
        inputs: Self::Inputs<Prim<B>>,
        extra: Self::Extra,
    ) -> Self::Inputs<Option<Prim<B>>>;
}

/// The node. It has no data, so an op type needs no `Debug` and no `Send`.
struct RecomputeNode<O>(PhantomData<fn() -> O>);

impl<O> fmt::Debug for RecomputeNode<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RecomputeNode")
    }
}

/// The shapes of the two outputs, from the results of the forward.
#[derive(Clone, Debug)]
struct OutputShapes {
    y: Shape,
    final_state: Shape,
}

/// What the node keeps from the forward for the backward.
#[derive(Clone, Debug)]
struct RecomputeState<B: Backend, E, const N: usize> {
    inputs: [Prim<B>; N],
    extra: E,
    shapes: OutputShapes,
}

impl<B, O, const N: usize> Backward<B, N> for RecomputeNode<O>
where
    B: Backend,
    O: RecomputeOp<B, N>,
{
    type State = RecomputeState<B, O::Extra, N>;

    fn backward(
        self,
        ops: Ops<Self::State, N>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let d_packed = grads.consume::<B>(&ops.node);
        let RecomputeState { inputs, extra, shapes } = ops.state;
        let (d_y, d_final_state) = split_outputs::<B>(d_packed, shapes);
        let d_inputs = O::backward(d_y, d_final_state, O::Inputs::from_array(inputs), extra);
        for (parent, d_input) in ops.parents.into_iter().zip(d_inputs.into_array()) {
            if let (Some(parent), Some(d_input)) = (parent, d_input) {
                grads.register::<B>(parent.id, d_input);
            }
        }
    }
}

/// Flatten the two outputs and concatenate them. Returns the 1-D tensor and
/// the two output shapes.
fn pack_outputs<B: Backend>(y: Prim<B>, final_state: Prim<B>) -> (Prim<B>, OutputShapes) {
    let shapes = OutputShapes { y: y.shape(), final_state: final_state.shape() };
    let flat_y = B::float_reshape(y, Shape::new([shapes.y.num_elements()]));
    let flat_final_state =
        B::float_reshape(final_state, Shape::new([shapes.final_state.num_elements()]));
    (B::float_cat(vec![flat_y, flat_final_state], 0), shapes)
}

/// The inverse of [`pack_outputs`]. The function is generic over the backend,
/// so one body serves the gradient (on `B`) and the output (on
/// `Autodiff<B, C>`).
fn split_outputs<B: Backend>(packed: Prim<B>, shapes: OutputShapes) -> (Prim<B>, Prim<B>) {
    let len_y = shapes.y.num_elements();
    let len_final_state = shapes.final_state.num_elements();
    let flat_y = B::float_slice(packed.clone(), &[s![0..len_y]]);
    let flat_final_state = B::float_slice(packed, &[s![len_y..len_y + len_final_state]]);
    (B::float_reshape(flat_y, shapes.y), B::float_reshape(flat_final_state, shapes.final_state))
}

/// Run the op `O` as one autodiff node, and return `y` and `final_state`.
#[allow(clippy::type_complexity)]
pub fn recompute_pair<B, C, O, const N: usize>(
    inputs: <O as RecomputeOp<B, N>>::Inputs<FloatTensor<Autodiff<B, C>>>,
    extra: O::Extra,
) -> (FloatTensor<Autodiff<B, C>>, FloatTensor<Autodiff<B, C>>)
where
    B: Backend,
    C: CheckpointStrategy,
    O: RecomputeOp<B, N>,
{
    let tensors = inputs.into_array();
    let guards = tensors.each_ref().map(|tensor| tensor.node());
    let prims: [Prim<B>; N] = tensors.map(|tensor| tensor.into_primitive());
    let forward_packed = |prims: [Prim<B>; N]| {
        let (y, final_state) = O::forward(O::Inputs::from_array(prims), &extra);
        pack_outputs::<B>(y, final_state)
    };
    let node = RecomputeNode::<O>(PhantomData);
    match <RecomputeNode<O> as Backward<B, N>>::prepare::<C>(node, guards)
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            let (packed, shapes) = forward_packed(prims.clone());
            let state = RecomputeState { inputs: prims, extra, shapes: shapes.clone() };
            split_outputs::<Autodiff<B, C>>(prep.finish(state, packed), shapes)
        }
        OpsKind::UnTracked(prep) => {
            let (packed, shapes) = forward_packed(prims);
            split_outputs::<Autodiff<B, C>>(prep.finish(packed), shapes)
        }
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
