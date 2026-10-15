//! Collection aliases for sets and maps keyed by [`AttackGraphNodeId`].
//!
//! Node ids are slotmap keys (not attacker-controlled), so the fast
//! non-cryptographic FxHash is safe here. It is also non-randomized, so
//! iteration order is deterministic for a given insertion sequence.

use maltoolbox_attackgraph::AttackGraphNodeId;

/// A set of attack graph node ids, hashed with FxHash.
pub type NodeSet = rustc_hash::FxHashSet<AttackGraphNodeId>;
/// A map keyed by attack graph node id, hashed with FxHash.
pub type NodeMap<V> = rustc_hash::FxHashMap<AttackGraphNodeId, V>;
/// Dense per-node storage for values every node has; indexed by slot, no hashing.
pub type NodeDenseMap<V> = slotmap::SecondaryMap<AttackGraphNodeId, V>;
