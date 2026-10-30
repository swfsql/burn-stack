# The code is block-agnostic, the docs can cite a family

burn-stack holds the containers around a block: layers, layer stacks,
networks, caches and optimizer plans. More than one family uses it:
burn-mamba and burn-deltanet. So the *code* of burn-stack is block-agnostic.
It has no family name and no shape assumption that only one family makes. An
item that only one family needs goes into the crate of that family.

The *docs* can name a family as a citation: the origin of a mechanism, a
reference implementation, or a consumer. For example, the H-Net header says
that the reference uses Mamba-2 outside and attention inside.

burn-mamba ADR-0002 records the same rule from the side of a consumer.

## Consequences

- An exception is rare. A mechanism that more than one family uses can have
  a setting in burn-stack that only some families need. The code names the
  setting after the mechanism, not after a family. Its doc says why it is in
  burn-stack.
- A consumer forwards every feature of burn-stack. The `backend-*` cfgs take
  effect where `impl_backend_ext_for_burn_backends!` expands, which is in the
  consumer crate. When only one side enables a backend, that backend loses
  its `BackendExt` impls, and no error shows it.
