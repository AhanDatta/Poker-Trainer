//! Three-player (`solver::multi`) solves on the GPU: the plan
//! ([`plan::MultiPlan`]) and the engine that runs it.

pub(crate) mod engine;
pub(crate) mod plan;
#[cfg(test)]
mod testutil;
