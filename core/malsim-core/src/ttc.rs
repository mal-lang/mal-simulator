//! Rust port of `python/malsim/mal_simulator/ttc_utils.py`'s `TTCDist` -
//! see `PORTING_NOTES.md` §5 Phase A2.
//!
//! Terminology kept close to the Python original (§2.7): `DistFunction`,
//! `Operation`, `TtcDist` (PascalCase per Rust naming conventions - same
//! concept as Python's `TTCDist`), `named_ttc_dist`, `expected_value`,
//! `sample_value`, `success_probability`, `attempt_ttc_with_effort`,
//! `attempt_bernoulli`.
//!
//! Sampling is statistically-, not bit-, equivalent to the Python/numpy/
//! scipy original - see `PORTING_NOTES.md` §2.1. `expected_value` and
//! `success_probability`, however, are closed-form and therefore exactly
//! reproducible - the two are not equally affected by the RNG-equivalence
//! caveat.
//!
//! Scope note: `default_ttc_dist`/`TTCDist.from_node`/`from_name`
//! (Python's graph-node-dependent helpers) are *not* ported here - they
//! belong to Phase A3, which ports the pieces of `compute_initial_graph_state`
//! that read `AttackGraphNode`. This module is deliberately independent of
//! `maltoolbox_attackgraph` so it's unit-testable on its own.

use std::fmt;

use rand::{Rng, RngExt};
use serde_json::{json, Value};
use statrs::distribution::{
    Bernoulli as StatrsBernoulli, Binomial as StatrsBinomial, ContinuousCDF, DiscreteCDF, Exp,
    Gamma, LogNormal, Uniform as StatrsUniform,
};
use statrs::statistics::Distribution as StatrsMean;

/// Mirrors Python's `DistFunction` enum - variant names are PascalCase per
/// Rust convention, but `as_str`/`from_str` use the exact same string
/// values as Python's `DistFunction.value` (the wire format used by
/// `to_dict`/`from_dict`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistFunction {
    Bernoulli,
    Exponential,
    Binomial,
    Gamma,
    LogNormal,
    Uniform,
}

impl DistFunction {
    pub fn as_str(self) -> &'static str {
        match self {
            DistFunction::Bernoulli => "Bernoulli",
            DistFunction::Exponential => "Exponential",
            DistFunction::Binomial => "Binomial",
            DistFunction::Gamma => "Gamma",
            DistFunction::LogNormal => "LogNormal",
            DistFunction::Uniform => "Uniform",
        }
    }

    /// Number of distribution parameters `TtcDist::new` expects in `args`.
    fn expected_arg_count(self) -> usize {
        match self {
            DistFunction::Bernoulli => 1,
            DistFunction::Exponential => 1,
            DistFunction::Binomial => 2,
            DistFunction::Gamma => 2,
            DistFunction::LogNormal => 2,
            DistFunction::Uniform => 2,
        }
    }
}

impl std::str::FromStr for DistFunction {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Bernoulli" => Ok(DistFunction::Bernoulli),
            "Exponential" => Ok(DistFunction::Exponential),
            "Binomial" => Ok(DistFunction::Binomial),
            "Gamma" => Ok(DistFunction::Gamma),
            "LogNormal" => Ok(DistFunction::LogNormal),
            "Uniform" => Ok(DistFunction::Uniform),
            _ => Err(()),
        }
    }
}

/// Mirrors Python's `Operation` enum (combination operator between two
/// `TtcDist`s). `as_str`/`FromStr` match Python's `Operation.value`
/// strings (lowercase).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Exponentiation,
}

impl Operation {
    pub fn as_str(self) -> &'static str {
        match self {
            Operation::Addition => "addition",
            Operation::Subtraction => "subtraction",
            Operation::Multiplication => "multiplication",
            Operation::Division => "division",
            Operation::Exponentiation => "exponentiation",
        }
    }
}

impl std::str::FromStr for Operation {
    type Err = ();

    /// Case-insensitive, matching Python's `Operation[ttc_dict['type'].upper()]`
    /// lookup-by-member-name (the wire format itself is always lowercase,
    /// via `to_dict`, but `from_dict` normalizes case the same way Python
    /// does before matching).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "addition" => Ok(Operation::Addition),
            "subtraction" => Ok(Operation::Subtraction),
            "multiplication" => Ok(Operation::Multiplication),
            "division" => Ok(Operation::Division),
            "exponentiation" => Ok(Operation::Exponentiation),
            _ => Err(()),
        }
    }
}

pub fn perform_operation(op: Operation, a: f64, b: f64) -> f64 {
    match op {
        Operation::Addition => a + b,
        Operation::Subtraction => a - b,
        Operation::Multiplication => a * b,
        Operation::Division => a / b,
        Operation::Exponentiation => a.powf(b),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TtcDistError {
    WrongArgCount {
        function: DistFunction,
        expected: usize,
        actual: usize,
    },
    InvalidDistributionParams(String),
    UnknownDistFunction(String),
    UnknownOperation(String),
    InvalidShape(String),
}

impl fmt::Display for TtcDistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TtcDistError::WrongArgCount {
                function,
                expected,
                actual,
            } => write!(
                f,
                "{} expects {expected} argument(s), got {actual}",
                function.as_str()
            ),
            TtcDistError::InvalidDistributionParams(msg) => {
                write!(f, "invalid distribution parameters: {msg}")
            }
            TtcDistError::UnknownDistFunction(name) => {
                write!(f, "unknown distribution function name '{name}'")
            }
            TtcDistError::UnknownOperation(name) => {
                write!(f, "unknown operation '{name}'")
            }
            TtcDistError::InvalidShape(msg) => write!(f, "invalid TtcDist dict shape: {msg}"),
        }
    }
}

impl std::error::Error for TtcDistError {}

/// Rust port of Python's `TTCDist`. Unlike the Python original, this does
/// not eagerly store a constructed `scipy`-equivalent distribution object -
/// `statrs` distribution values are cheap to (re)build from `args` on each
/// call, so they're constructed on demand in each method instead
/// (`TtcDist::new` still validates `args` eagerly, for the same fail-fast
/// behavior as Python's constructor).
#[derive(Debug, Clone, PartialEq)]
pub struct TtcDist {
    pub function: DistFunction,
    pub args: Vec<f64>,
    pub combine_with: Option<Box<TtcDist>>,
    pub combine_op: Option<Operation>,
}

impl TtcDist {
    pub fn new(function: DistFunction, args: Vec<f64>) -> Result<TtcDist, TtcDistError> {
        let expected = function.expected_arg_count();
        if args.len() != expected {
            return Err(TtcDistError::WrongArgCount {
                function,
                expected,
                actual: args.len(),
            });
        }
        let dist = TtcDist {
            function,
            args,
            combine_with: None,
            combine_op: None,
        };
        // Validate params eagerly (mirrors Python's constructor, which
        // eagerly builds `self.dist` and raises immediately on bad params).
        dist.validate()?;
        Ok(dist)
    }

    pub fn with_combine(mut self, combine_with: TtcDist, combine_op: Operation) -> TtcDist {
        self.combine_with = Some(Box::new(combine_with));
        self.combine_op = Some(combine_op);
        self
    }

    fn validate(&self) -> Result<(), TtcDistError> {
        match self.function {
            DistFunction::Bernoulli => {
                StatrsBernoulli::new(self.args[0])
                    .map_err(|e| TtcDistError::InvalidDistributionParams(e.to_string()))?;
            }
            DistFunction::Exponential => {
                Exp::new(self.args[0])
                    .map_err(|e| TtcDistError::InvalidDistributionParams(e.to_string()))?;
            }
            DistFunction::Binomial => {
                StatrsBinomial::new(self.args[1], self.args[0] as u64)
                    .map_err(|e| TtcDistError::InvalidDistributionParams(e.to_string()))?;
            }
            DistFunction::Gamma => {
                Gamma::new(self.args[0], 1.0 / self.args[1])
                    .map_err(|e| TtcDistError::InvalidDistributionParams(e.to_string()))?;
            }
            DistFunction::LogNormal => {
                LogNormal::new(self.args[0], self.args[1])
                    .map_err(|e| TtcDistError::InvalidDistributionParams(e.to_string()))?;
            }
            DistFunction::Uniform => {
                StatrsUniform::new(self.args[0], self.args[1])
                    .map_err(|e| TtcDistError::InvalidDistributionParams(e.to_string()))?;
            }
        }
        Ok(())
    }

    // These re-construct the underlying `statrs` distribution each call -
    // `args` was already validated in `new`, so these are expected to
    // never fail; see `validate` above.
    fn bernoulli(&self) -> StatrsBernoulli {
        StatrsBernoulli::new(self.args[0]).expect("validated in TtcDist::new")
    }
    fn exponential(&self) -> Exp {
        Exp::new(self.args[0]).expect("validated in TtcDist::new")
    }
    fn binomial(&self) -> StatrsBinomial {
        // Python: `n, p = args` (args[0] = n, args[1] = p); statrs takes
        // (p, n), the opposite order.
        StatrsBinomial::new(self.args[1], self.args[0] as u64).expect("validated in TtcDist::new")
    }
    fn gamma(&self) -> Gamma {
        // Python: `shape, scale = args`; statrs takes (shape, rate), where
        // rate = 1 / scale.
        Gamma::new(self.args[0], 1.0 / self.args[1]).expect("validated in TtcDist::new")
    }
    fn log_normal(&self) -> LogNormal {
        // Python: `mean, std = args` matches statrs's (location, scale).
        LogNormal::new(self.args[0], self.args[1]).expect("validated in TtcDist::new")
    }
    fn uniform(&self) -> StatrsUniform {
        // Python: `low, high = args` matches statrs's (min, max).
        StatrsUniform::new(self.args[0], self.args[1]).expect("validated in TtcDist::new")
    }

    /// Return the expected value of this `TtcDist`. Closed-form (no RNG) -
    /// exactly reproducible, unlike `sample_value`.
    pub fn expected_value(&self) -> f64 {
        let value = match self.function {
            // Expected value of Bernoulli always assumes a successful
            // trial for attack-step existence / defense initial state
            // purposes - not the distribution's real mean. Mirrors
            // Python's identically-worded special case.
            DistFunction::Bernoulli => 1.0,
            DistFunction::Exponential => self.exponential().mean().expect("exponential has mean"),
            DistFunction::Binomial => self.binomial().mean().expect("binomial has mean"),
            DistFunction::Gamma => self.gamma().mean().expect("gamma has mean"),
            DistFunction::LogNormal => self.log_normal().mean().expect("log-normal has mean"),
            DistFunction::Uniform => self.uniform().mean().expect("uniform has mean"),
        };
        match (&self.combine_with, self.combine_op) {
            (Some(rhs), Some(op)) => perform_operation(op, value, rhs.expected_value()),
            _ => value,
        }
    }

    /// Sample a value from this `TtcDist`. Statistically, not bit-,
    /// equivalent to Python's `sample_value` - see module docs / §2.1.
    pub fn sample_value(&self, rng: &mut impl Rng) -> f64 {
        let value = match self.function {
            // Mirrors Python: sampled Bernoulli value is always 1.0
            // (existence/initial-state semantics, not a real draw).
            DistFunction::Bernoulli => 1.0,
            DistFunction::Exponential => rng.sample(self.exponential()),
            DistFunction::Binomial => rng.sample(self.binomial()),
            DistFunction::Gamma => rng.sample(self.gamma()),
            DistFunction::LogNormal => rng.sample(self.log_normal()),
            DistFunction::Uniform => rng.sample(self.uniform()),
        };
        match (&self.combine_with, self.combine_op) {
            (Some(rhs), Some(op)) => perform_operation(op, value, rhs.sample_value(rng)),
            _ => value,
        }
    }

    /// The probability of success with the given `effort` (previous
    /// attempts) - the base distribution's CDF at `effort`. Note this
    /// reads only `self`'s own distribution, never `combine_with` -
    /// mirrors Python's `success_probability` exactly (not a bug carried
    /// over silently: Python's `dist.cdf(effort)` likewise never touches
    /// `self.combine_with`).
    pub fn success_probability(&self, effort: u64) -> f64 {
        match self.function {
            DistFunction::Bernoulli => self.bernoulli().cdf(effort),
            DistFunction::Exponential => self.exponential().cdf(effort as f64),
            DistFunction::Binomial => self.binomial().cdf(effort),
            DistFunction::Gamma => self.gamma().cdf(effort as f64),
            DistFunction::LogNormal => self.log_normal().cdf(effort as f64),
            DistFunction::Uniform => self.uniform().cdf(effort as f64),
        }
    }

    /// Attempt to compromise a step by sampling a success probability
    /// proportional to the TTC distribution, given previous attempts.
    /// Ignores Bernoullis (see `success_probability`).
    pub fn attempt_ttc_with_effort(&self, effort: u64, rng: &mut impl Rng) -> bool {
        self.success_probability(effort) > rng.random::<f64>()
    }

    /// Attempt bernoulli from a TTC distribution: if `self` is a
    /// Bernoulli, sample it directly; otherwise dig into `combine_with`
    /// (mirrors Python's recursive descent); if there's nothing left to
    /// dig into, the attempt is unconditionally successful.
    pub fn attempt_bernoulli(&self, rng: &mut impl Rng) -> bool {
        if self.function == DistFunction::Bernoulli {
            let threshold = self.args[0];
            rng.random::<f64>() <= threshold
        } else if let Some(rhs) = &self.combine_with {
            rhs.attempt_bernoulli(rng)
        } else {
            true
        }
    }

    /// Parse a `TtcDist` from the same JSON-ish dict shape Python's
    /// `TTCDist.to_dict`/`from_dict` use - either a named/simple
    /// `{"name": ..., "arguments": [...], "type": "function"}` leaf, or a
    /// `{"lhs": ..., "rhs": ..., "type": <operation>}` combination node.
    pub fn from_dict(value: &Value) -> Result<TtcDist, TtcDistError> {
        let obj = value
            .as_object()
            .ok_or_else(|| TtcDistError::InvalidShape("expected a JSON object".to_string()))?;

        if let Some(name) = obj.get("name").and_then(Value::as_str) {
            if let Some(named) = named_ttc_dist(name) {
                return Ok(named);
            }
            let function: DistFunction = name
                .parse()
                .map_err(|()| TtcDistError::UnknownDistFunction(name.to_string()))?;
            let args = obj
                .get("arguments")
                .and_then(Value::as_array)
                .ok_or_else(|| TtcDistError::InvalidShape("missing 'arguments' array".to_string()))?
                .iter()
                .map(|v| {
                    v.as_f64().ok_or_else(|| {
                        TtcDistError::InvalidShape("non-numeric argument".to_string())
                    })
                })
                .collect::<Result<Vec<f64>, _>>()?;
            return TtcDist::new(function, args);
        }

        let lhs_value = obj
            .get("lhs")
            .ok_or_else(|| TtcDistError::InvalidShape("missing 'lhs'".to_string()))?;
        let rhs_value = obj
            .get("rhs")
            .ok_or_else(|| TtcDistError::InvalidShape("missing 'rhs'".to_string()))?;
        let op_name = obj
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| TtcDistError::InvalidShape("missing 'type'".to_string()))?;

        let mut lhs = TtcDist::from_dict(lhs_value)?;
        let op: Operation = op_name
            .parse()
            .map_err(|()| TtcDistError::UnknownOperation(op_name.to_string()))?;
        let rhs = TtcDist::from_dict(rhs_value)?;
        lhs.combine_op = Some(op);
        lhs.combine_with = Some(Box::new(rhs));
        Ok(lhs)
    }

    /// Inverse of `from_dict` - produces the same dict shape Python's
    /// `to_dict` does.
    pub fn to_dict(&self) -> Value {
        let base = json!({
            "name": self.function.as_str(),
            "arguments": self.args,
            "type": "function",
        });
        match (&self.combine_with, self.combine_op) {
            (Some(rhs), Some(op)) => json!({
                "lhs": base,
                "rhs": rhs.to_dict(),
                "type": op.as_str(),
            }),
            _ => base,
        }
    }
}

/// Mirrors Python's `named_ttc_dists` dict - the MAL-defined named TTC
/// distributions. Returns a fresh `TtcDist` (cheap: no precomputed
/// distribution object is cached on `TtcDist` itself) rather than a
/// reference into a shared static, since the Python original's per-entry
/// identity is not otherwise observable (`TTCDist.__eq__` is structural).
pub fn named_ttc_dist(name: &str) -> Option<TtcDist> {
    let simple = |function: DistFunction, args: &[f64]| {
        TtcDist::new(function, args.to_vec()).expect("named_ttc_dist args are hand-verified valid")
    };

    Some(match name {
        "EasyAndUncertain" => simple(DistFunction::Bernoulli, &[0.5]),
        "HardAndUncertain" => simple(DistFunction::Exponential, &[0.1]).with_combine(
            simple(DistFunction::Bernoulli, &[0.5]),
            Operation::Multiplication,
        ),
        "VeryHardAndUncertain" => simple(DistFunction::Exponential, &[0.01]).with_combine(
            simple(DistFunction::Bernoulli, &[0.5]),
            Operation::Multiplication,
        ),
        "EasyAndCertain" => simple(DistFunction::Exponential, &[1.0]),
        "HardAndCertain" => simple(DistFunction::Exponential, &[0.1]),
        "VeryHardAndCertain" => simple(DistFunction::Exponential, &[0.01]),
        "Enabled" => simple(DistFunction::Bernoulli, &[1.0]),
        "Instant" => simple(DistFunction::Bernoulli, &[1.0]),
        "Disabled" => simple(DistFunction::Bernoulli, &[0.0]),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }

    fn dist(function: DistFunction, args: &[f64]) -> TtcDist {
        TtcDist::new(function, args.to_vec()).unwrap()
    }

    // --- expected_value: one per DistFunction variant, hand-computed ---

    #[test]
    fn expected_value_bernoulli_is_always_one() {
        approx(dist(DistFunction::Bernoulli, &[0.5]).expected_value(), 1.0);
        approx(dist(DistFunction::Bernoulli, &[0.0]).expected_value(), 1.0);
        approx(dist(DistFunction::Bernoulli, &[1.0]).expected_value(), 1.0);
    }

    #[test]
    fn expected_value_exponential_is_one_over_rate() {
        approx(
            dist(DistFunction::Exponential, &[0.1]).expected_value(),
            10.0,
        );
        approx(
            dist(DistFunction::Exponential, &[1.0]).expected_value(),
            1.0,
        );
        approx(
            dist(DistFunction::Exponential, &[0.01]).expected_value(),
            100.0,
        );
    }

    #[test]
    fn expected_value_binomial_is_n_times_p() {
        approx(
            dist(DistFunction::Binomial, &[10.0, 0.1]).expected_value(),
            1.0,
        );
    }

    #[test]
    fn expected_value_gamma_is_shape_times_scale() {
        approx(dist(DistFunction::Gamma, &[1.0, 0.1]).expected_value(), 0.1);
    }

    #[test]
    fn expected_value_log_normal_matches_scipy_formula() {
        let expected = (1.0_f64 + 0.1_f64.powi(2) / 2.0).exp();
        approx(
            dist(DistFunction::LogNormal, &[1.0, 0.1]).expected_value(),
            expected,
        );
    }

    #[test]
    fn expected_value_uniform_is_midpoint() {
        approx(
            dist(DistFunction::Uniform, &[1.0, 10.0]).expected_value(),
            5.5,
        );
    }

    // --- combine_with/combine_op composition - ported 1:1 from
    // test_ttc_utils.py::test_all_ttc_distributions, since expected_value
    // is deterministic (no RNG) and therefore portable bit-for-bit, unlike
    // sample_value. ---

    #[test]
    fn combine_multiplication_hard_and_uncertain() {
        let d = dist(DistFunction::Exponential, &[0.1]).with_combine(
            dist(DistFunction::Bernoulli, &[0.5]),
            Operation::Multiplication,
        );
        approx(d.expected_value(), 10.0);
    }

    #[test]
    fn combine_addition_binomial_plus_exponential() {
        let d = dist(DistFunction::Binomial, &[10.0, 0.1])
            .with_combine(dist(DistFunction::Exponential, &[0.1]), Operation::Addition);
        let expected = (10.0 * 0.1) + (1.0 / 0.1);
        approx(d.expected_value(), expected);
    }

    #[test]
    fn combine_subtraction_gamma_minus_binomial() {
        let d = dist(DistFunction::Gamma, &[1.0, 0.1]).with_combine(
            dist(DistFunction::Binomial, &[10.0, 0.1]),
            Operation::Subtraction,
        );
        let expected = (1.0 * 0.1) - (10.0 * 0.1);
        approx(d.expected_value(), expected);
    }

    #[test]
    fn combine_multiplication_log_normal_times_gamma() {
        let d = dist(DistFunction::LogNormal, &[1.0, 0.1]).with_combine(
            dist(DistFunction::Gamma, &[1.0, 0.1]),
            Operation::Multiplication,
        );
        let lognorm_mean = (1.0_f64 + 0.1_f64.powi(2) / 2.0).exp();
        let gamma_mean = 1.0 * 0.1;
        approx(d.expected_value(), lognorm_mean * gamma_mean);
    }

    #[test]
    fn combine_division_uniform_over_log_normal() {
        let d = dist(DistFunction::Uniform, &[1.0, 10.0]).with_combine(
            dist(DistFunction::LogNormal, &[1.0, 0.1]),
            Operation::Division,
        );
        let uniform_mean = (1.0 + 10.0) / 2.0;
        let lognorm_mean = (1.0_f64 + 0.1_f64.powi(2) / 2.0).exp();
        approx(d.expected_value(), uniform_mean / lognorm_mean);
    }

    #[test]
    fn combine_multiplication_exponential_times_uniform() {
        let d = dist(DistFunction::Exponential, &[0.5]).with_combine(
            dist(DistFunction::Uniform, &[1.0, 10.0]),
            Operation::Multiplication,
        );
        let exp_mean = 1.0 / 0.5;
        let uniform_mean = (1.0 + 10.0) / 2.0;
        approx(d.expected_value(), exp_mean * uniform_mean);
    }

    // --- success_probability / degenerate-distribution checks, as used
    // by (the not-yet-ported) get_pre_enabled_defenses ---

    #[test]
    fn success_probability_degenerate_bernoulli() {
        // 'Disabled': Bernoulli(0.0) -> cdf(0) = 1 - p = 1.0 ("always
        // succeeds", i.e. not pre-enabled in A3's caller).
        approx(
            named_ttc_dist("Disabled").unwrap().success_probability(0),
            1.0,
        );
        // 'Enabled'/'Instant': Bernoulli(1.0) -> cdf(0) = 1 - p = 0.0
        // ("never succeeds" at effort 0, i.e. pre-enabled in A3's caller).
        approx(
            named_ttc_dist("Enabled").unwrap().success_probability(0),
            0.0,
        );
    }

    #[test]
    fn success_probability_monotonic_with_effort() {
        let d = dist(DistFunction::Exponential, &[0.01]);
        let low = d.success_probability(1);
        let high = d.success_probability(500);
        assert!(low < high, "{low} should be < {high}");
        approx(d.success_probability(0), 0.0);
    }

    #[test]
    fn success_probability_ignores_combine_with() {
        // success_probability reads only self's own distribution, never
        // combine_with - mirrors Python's dist.cdf(effort) exactly.
        let plain = dist(DistFunction::Exponential, &[0.1]);
        let combined = dist(DistFunction::Exponential, &[0.1]).with_combine(
            dist(DistFunction::Bernoulli, &[0.5]),
            Operation::Multiplication,
        );
        approx(
            plain.success_probability(5),
            combined.success_probability(5),
        );
    }

    // --- attempt_bernoulli / attempt_ttc_with_effort: RNG-touching, so
    // only structural/statistical properties are asserted, not exact
    // sampled values pinned to one seed - see §2.1. ---

    #[test]
    fn attempt_bernoulli_degenerate_cases_are_deterministic() {
        let mut rng = StdRng::seed_from_u64(10);
        let always_true = dist(DistFunction::Bernoulli, &[1.0]);
        for _ in 0..100 {
            assert!(always_true.attempt_bernoulli(&mut rng));
        }
    }

    #[test]
    fn attempt_bernoulli_uncertain_gives_both_outcomes() {
        let mut rng = StdRng::seed_from_u64(10);
        let uncertain = dist(DistFunction::Bernoulli, &[0.5]);
        let mut saw_true = false;
        let mut saw_false = false;
        for _ in 0..200 {
            if uncertain.attempt_bernoulli(&mut rng) {
                saw_true = true;
            } else {
                saw_false = true;
            }
        }
        assert!(saw_true && saw_false);
    }

    #[test]
    fn attempt_bernoulli_digs_into_combine_with() {
        let mut rng = StdRng::seed_from_u64(1);
        // HardAndUncertain: Exponential(0.1) * Bernoulli(0.5) - attempt_bernoulli
        // should recurse into the Bernoulli, not the Exponential.
        let d = named_ttc_dist("HardAndUncertain").unwrap();
        let mut saw_true = false;
        let mut saw_false = false;
        for _ in 0..200 {
            if d.attempt_bernoulli(&mut rng) {
                saw_true = true;
            } else {
                saw_false = true;
            }
        }
        assert!(saw_true && saw_false);
    }

    #[test]
    fn attempt_ttc_with_effort_success_rate_matches_probability() {
        let mut rng = StdRng::seed_from_u64(42);
        let d = dist(DistFunction::Exponential, &[0.1]);
        let effort = 10;
        let expected_p = d.success_probability(effort);
        let trials = 5000;
        let successes = (0..trials)
            .filter(|_| d.attempt_ttc_with_effort(effort, &mut rng))
            .count();
        let observed_p = successes as f64 / trials as f64;
        assert!(
            (observed_p - expected_p).abs() < 0.05,
            "observed {observed_p}, expected ~{expected_p}"
        );
    }

    // --- from_dict / to_dict ---

    #[test]
    fn to_dict_from_dict_roundtrip_simple() {
        let d = dist(DistFunction::Exponential, &[0.1]);
        let round_tripped = TtcDist::from_dict(&d.to_dict()).unwrap();
        assert_eq!(d, round_tripped);
    }

    #[test]
    fn to_dict_from_dict_roundtrip_combined() {
        let d = dist(DistFunction::Gamma, &[1.0, 0.1]).with_combine(
            dist(DistFunction::Binomial, &[10.0, 0.1]),
            Operation::Subtraction,
        );
        let round_tripped = TtcDist::from_dict(&d.to_dict()).unwrap();
        assert_eq!(d, round_tripped);
    }

    #[test]
    fn from_dict_resolves_named_dist_by_name() {
        let value = json!({"name": "Instant"});
        let parsed = TtcDist::from_dict(&value).unwrap();
        assert_eq!(parsed, named_ttc_dist("Instant").unwrap());
    }

    #[test]
    fn from_dict_rejects_unknown_function_name() {
        let value = json!({"name": "NotARealDistribution", "arguments": [1.0]});
        assert!(matches!(
            TtcDist::from_dict(&value),
            Err(TtcDistError::UnknownDistFunction(_))
        ));
    }

    #[test]
    fn from_dict_rejects_unknown_operation() {
        let value = json!({
            "lhs": {"name": "Exponential", "arguments": [0.1]},
            "rhs": {"name": "Bernoulli", "arguments": [0.5]},
            "type": "not_a_real_op",
        });
        assert!(matches!(
            TtcDist::from_dict(&value),
            Err(TtcDistError::UnknownOperation(_))
        ));
    }

    #[test]
    fn new_rejects_wrong_arg_count() {
        assert!(matches!(
            TtcDist::new(DistFunction::Exponential, vec![0.1, 0.2]),
            Err(TtcDistError::WrongArgCount { .. })
        ));
    }

    #[test]
    fn named_ttc_dist_matches_expected_structure() {
        let hard_and_uncertain = named_ttc_dist("HardAndUncertain").unwrap();
        assert_eq!(hard_and_uncertain.function, DistFunction::Exponential);
        assert_eq!(hard_and_uncertain.args, vec![0.1]);
        assert_eq!(
            hard_and_uncertain.combine_op,
            Some(Operation::Multiplication)
        );
        let combine_with = hard_and_uncertain.combine_with.as_ref().unwrap();
        assert_eq!(combine_with.function, DistFunction::Bernoulli);
        assert_eq!(combine_with.args, vec![0.5]);

        assert!(named_ttc_dist("NotANamedDist").is_none());
    }
}
