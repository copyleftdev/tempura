/// State trait — candidate solution representation.
///
/// # Contract
/// - Must be Clone (needed for undo-on-reject in SA)
/// - Must be Debug (observability requirement)
///
/// # Design rationale (Matsakis: ownership)
/// State is owned by the annealer. Move operators borrow it immutably to
/// produce a new candidate. On rejection, the candidate is dropped.
/// On acceptance, the candidate replaces the current state.
///
/// For large states, consider in-place mutation with undo (see `ReversibleMove`).
use core::fmt::Debug;

/// Marker trait for solution states.
///
/// Any type that is `Clone + Debug` can serve as a state.
/// This is intentionally minimal — Tempura imposes no structure on
/// the solution representation.
pub trait State: Clone + Debug {}

/// Blanket implementation: anything Clone + Debug is a State.
impl<T: Clone + Debug> State for T {}
