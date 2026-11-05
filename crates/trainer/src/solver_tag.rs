//! Which solver produced a review or study result (shown with it and kept
//! in saved sessions).

use serde::{Deserialize, Serialize};

/// The solver behind a result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolvedBy {
    /// The CPU (heads-up or 3-way). Also what results saved before solvers
    /// were recorded read as.
    #[default]
    Cpu,
    /// The GPU with double-single terminal sums (the GPU default).
    GpuFast,
    /// The GPU with f64 terminal sums (bit-identical to the CPU).
    GpuExact,
}

impl SolvedBy {
    pub fn label(self) -> &'static str {
        match self {
            SolvedBy::Cpu => "CPU",
            SolvedBy::GpuFast => "GPU (fast)",
            SolvedBy::GpuExact => "GPU (exact)",
        }
    }
}

impl From<solver::SolvedOn> for SolvedBy {
    fn from(s: solver::SolvedOn) -> SolvedBy {
        match s {
            solver::SolvedOn::Cpu => SolvedBy::Cpu,
            solver::SolvedOn::GpuFast => SolvedBy::GpuFast,
            solver::SolvedOn::GpuExact => SolvedBy::GpuExact,
        }
    }
}

/// The solver behind a result, and why the GPU wasn't used when the
/// settings asked for it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverTag {
    pub by: SolvedBy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_fallback: Option<String>,
}

impl SolverTag {
    /// "GPU (fast)", "CPU", or "CPU (GPU not used: …)".
    pub fn describe(&self) -> String {
        match (&self.gpu_fallback, self.by) {
            (Some(why), SolvedBy::Cpu) => format!("CPU (GPU not used: {why})"),
            (_, by) => by.label().to_string(),
        }
    }
}

/// What a finished solve reports besides the solver itself.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SolveSummary {
    /// Exploitability (heads-up) or convergence (3-way), % of the pot.
    pub exploitability_pct: f64,
    pub solver: SolverTag,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describes_each_solver() {
        let tag = |by, why: Option<&str>| SolverTag { by, gpu_fallback: why.map(str::to_string) };
        assert_eq!(tag(SolvedBy::Cpu, None).describe(), "CPU");
        assert_eq!(tag(SolvedBy::GpuFast, None).describe(), "GPU (fast)");
        assert_eq!(tag(SolvedBy::GpuExact, None).describe(), "GPU (exact)");
        assert_eq!(tag(SolvedBy::Cpu, Some("no NVIDIA driver")).describe(), "CPU (GPU not used: no NVIDIA driver)");
    }

    #[test]
    fn maps_every_solver_backend() {
        for (on, by) in [
            (solver::SolvedOn::Cpu, SolvedBy::Cpu),
            (solver::SolvedOn::GpuFast, SolvedBy::GpuFast),
            (solver::SolvedOn::GpuExact, SolvedBy::GpuExact),
        ] {
            assert_eq!(SolvedBy::from(on), by);
        }
    }

    /// Tags survive a save; results saved before tags existed read as CPU.
    #[test]
    fn round_trips_and_defaults_to_cpu() {
        let tag = SolverTag { by: SolvedBy::Cpu, gpu_fallback: Some("out of memory".into()) };
        let text = serde_json::to_string(&tag).unwrap();
        assert_eq!(serde_json::from_str::<SolverTag>(&text).unwrap(), tag);
        let fast = SolverTag { by: SolvedBy::GpuFast, gpu_fallback: None };
        let text = serde_json::to_string(&fast).unwrap();
        assert_eq!(text, r#"{"by":"gpu_fast"}"#);
        assert_eq!(serde_json::from_str::<SolverTag>("{}").unwrap(), SolverTag::default());

        #[derive(Deserialize)]
        struct Old {
            #[serde(default)]
            solver: SolverTag,
        }
        let old: Old = serde_json::from_str("{}").unwrap();
        assert_eq!(old.solver.by, SolvedBy::Cpu);
    }
}
