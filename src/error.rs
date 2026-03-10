//! Error types for the Tempura annealing framework.
//!
//! Library code returns `Result<T, AnnealError>` instead of panicking
//! so that callers can handle configuration errors gracefully.

/// Errors that can occur when configuring or running an annealing algorithm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnnealError {
    /// A required builder field was not set.
    MissingField {
        /// Name of the missing field.
        field: &'static str,
    },
    /// A parameter was out of its valid range.
    InvalidParameter {
        /// Name of the invalid parameter.
        name: &'static str,
        /// Description of the constraint that was violated.
        reason: &'static str,
    },
}

impl core::fmt::Display for AnnealError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingField { field } => {
                write!(f, "missing required field: {field}")
            }
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter '{name}': {reason}")
            }
        }
    }
}

impl std::error::Error for AnnealError {}
