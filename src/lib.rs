//! # terminal-entropy-harness
//!
//! A standalone extraction of the Intelligent Terminal's Verification Entropy
//! module. Tracks the ratio of edits to test runs and computes a
//! "verification entropy" metric that measures how much latent bug risk
//! has accumulated since the last verification cycle.
//!
//! ## Model
//!
//! ```text
//! E = 1 - exp(-α · edits_since_last_test / L)
//! ```
//!
//! Where:
//! - `α` is a scaling factor (default 0.005)
//! - `L` is a reference "lines per test unit" (default 3.0)
//! - The result is clamped to [0.0, 1.0]
//!
//! ## Usage
//!
//! ```rust
//! use terminal_entropy_harness::{VerificationEntropy, EntropyLevel};
//!
//! let mut tracker = VerificationEntropy::new();
//! tracker.record_edit(50);
//! assert!(tracker.entropy() > 0.0);
//! assert_eq!(tracker.current_level(), EntropyLevel::Low);
//!
//! tracker.record_test();
//! assert_eq!(tracker.entropy(), 0.0);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// Threshold levels for verification entropy warnings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntropyLevel {
    /// Green: recently verified, low risk.
    Low,
    /// Yellow: accumulating entropy, moderate risk.
    Medium,
    /// Orange: significant unverified changes, get ready to test.
    High,
    /// Red: conservation of verification entropy guarantees bugs are coming.
    Critical,
}

impl fmt::Display for EntropyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntropyLevel::Low => write!(f, "low"),
            EntropyLevel::Medium => write!(f, "medium"),
            EntropyLevel::High => write!(f, "high"),
            EntropyLevel::Critical => write!(f, "critical"),
        }
    }
}

/// An event emitted by the entropy tracker when thresholds are crossed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationEvent {
    /// The current entropy value when the event was emitted.
    pub entropy: f64,
    /// Human-readable severity level.
    pub level: EntropyLevel,
    /// How many lines have been edited since the last test.
    pub edits_since_last_test: u64,
    /// Total lines edited across all sessions.
    pub total_lines_edited: u64,
    /// Total test commands run across all sessions.
    pub total_tests_run: u64,
    /// A message explaining the current state.
    pub message: String,
}

/// Tracks the edit-to-test ratio and computes verification entropy.
///
/// Serialize/deserialize to persist across sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationEntropy {
    /// The alpha decay factor for the entropy formula.
    alpha: f64,
    /// Reference "lines per test unit."
    lines_per_test_unit: f64,
    /// Edits (in lines) since the last test command was detected.
    edits_since_last_test: u64,
    /// Running total of lines edited.
    total_lines_edited: u64,
    /// Running total of test commands run.
    total_tests_run: u64,
    /// Thresholds for each entropy level.
    medium_threshold: f64,
    high_threshold: f64,
    critical_threshold: f64,
}

impl VerificationEntropy {
    /// Create a new entropy tracker with default parameters.
    ///
    /// Defaults:
    /// - α = 0.005
    /// - lines_per_test_unit = 3.0
    /// - Medium threshold = 0.30
    /// - High threshold = 0.60
    /// - Critical threshold = 0.80
    pub fn new() -> Self {
        Self {
            alpha: 0.005,
            lines_per_test_unit: 3.0,
            edits_since_last_test: 0,
            total_lines_edited: 0,
            total_tests_run: 0,
            medium_threshold: 0.30,
            high_threshold: 0.60,
            critical_threshold: 0.80,
        }
    }

    /// Create a new entropy tracker with custom parameters.
    ///
    /// - `alpha`: decay factor for the entropy formula (higher = faster growth)
    /// - `lines_per_test_unit`: reference lines-per-test scaling
    /// - `medium_threshold`: entropy threshold for Medium level
    /// - `high_threshold`: entropy threshold for High level
    /// - `critical_threshold`: entropy threshold for Critical level
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        alpha: f64,
        lines_per_test_unit: f64,
        medium_threshold: f64,
        high_threshold: f64,
        critical_threshold: f64,
    ) -> Self {
        Self {
            alpha,
            lines_per_test_unit,
            edits_since_last_test: 0,
            total_lines_edited: 0,
            total_tests_run: 0,
            medium_threshold,
            high_threshold,
            critical_threshold,
        }
    }

    /// Record that `lines` lines of code were edited. Returns an event if
    /// the entropy level crossed a warning threshold (High or Critical).
    pub fn record_edit(&mut self, lines: u64) -> Option<VerificationEvent> {
        self.edits_since_last_test += lines;
        self.total_lines_edited += lines;

        let entropy_before = self.entropy();
        let level_before = self.level(entropy_before);

        if level_before == EntropyLevel::High || level_before == EntropyLevel::Critical {
            Some(VerificationEvent {
                entropy: entropy_before,
                level: level_before,
                edits_since_last_test: self.edits_since_last_test,
                total_lines_edited: self.total_lines_edited,
                total_tests_run: self.total_tests_run,
                message: self.build_message(),
            })
        } else {
            None
        }
    }

    /// Record that a test command was executed. Resets the
    /// `edits_since_last_test` counter and returns a discharge event.
    pub fn record_test(&mut self) -> VerificationEvent {
        let entropy_before = self.entropy();
        let level_after = EntropyLevel::Low;

        self.total_tests_run += 1;
        self.edits_since_last_test = 0;

        let entropy_after = self.entropy();
        let message = if entropy_before > 0.5 {
            format!(
                "Testing discharged entropy from {:.0}% to {:.0}%",
                entropy_before * 100.0,
                entropy_after * 100.0,
            )
        } else {
            "Fresh test run — entropy reset.".to_string()
        };

        VerificationEvent {
            entropy: entropy_after,
            level: level_after,
            edits_since_last_test: 0,
            total_lines_edited: self.total_lines_edited,
            total_tests_run: self.total_tests_run,
            message,
        }
    }

    /// Record a batch of edits and return all events that crossed thresholds.
    pub fn record_edits(&mut self, lines: u64) -> Vec<VerificationEvent> {
        let mut events = Vec::new();
        for _ in 0..lines {
            if let Some(event) = self.record_edit(1) {
                events.push(event);
            }
        }
        events
    }

    /// Compute the current entropy value.
    ///
    /// `E = 1 - exp(-α · edits_since_last_test / L)`
    ///
    /// Result is clamped to [0.0, 1.0].
    pub fn entropy(&self) -> f64 {
        let effective = self.edits_since_last_test as f64 / self.lines_per_test_unit;
        let raw = 1.0 - (-self.alpha * effective).exp();
        raw.clamp(0.0, 1.0)
    }

    /// The raw count of edits since the last test.
    pub fn edits_since_last_test(&self) -> u64 {
        self.edits_since_last_test
    }

    /// Total lines edited across all sessions.
    pub fn total_lines_edited(&self) -> u64 {
        self.total_lines_edited
    }

    /// Total test commands run across all sessions.
    pub fn total_tests_run(&self) -> u64 {
        self.total_tests_run
    }

    /// The ratio of total edits to total tests.
    ///
    /// Returns `f64::MAX` if no tests have been run.
    pub fn edit_test_ratio(&self) -> f64 {
        if self.total_tests_run == 0 {
            f64::MAX
        } else {
            self.total_lines_edited as f64 / self.total_tests_run as f64
        }
    }

    /// Classify an entropy value into a severity level.
    pub fn level(&self, entropy: f64) -> EntropyLevel {
        if entropy >= self.critical_threshold {
            EntropyLevel::Critical
        } else if entropy >= self.high_threshold {
            EntropyLevel::High
        } else if entropy >= self.medium_threshold {
            EntropyLevel::Medium
        } else {
            EntropyLevel::Low
        }
    }

    /// Get the current entropy level.
    pub fn current_level(&self) -> EntropyLevel {
        self.level(self.entropy())
    }

    /// Build a human-readable message for the given entropy level.
    pub fn build_message(&self) -> String {
        let level = self.current_level();
        let pct = (self.entropy() * 100.0).round() as u64;
        match level {
            EntropyLevel::Critical => {
                format!(
                    "⚠ CONSERVATION OF VERIFICATION ENTROPY: {} lines edited without testing \
                     ({pct}%). Bugs are coming. Run tests now.",
                    self.edits_since_last_test,
                )
            }
            EntropyLevel::High => {
                format!(
                    "⚠ {} lines edited without testing ({pct}%). Verification entropy says latent \
                     bugs are accumulating. Consider running tests soon.",
                    self.edits_since_last_test,
                )
            }
            EntropyLevel::Medium => {
                format!(
                    "{} lines edited without testing ({pct}%). Entropy is building — test \
                     when convenient.",
                    self.edits_since_last_test,
                )
            }
            EntropyLevel::Low => {
                format!(
                    "Good: only {} lines since last test.",
                    self.edits_since_last_test
                )
            }
        }
    }

    /// Get a short status bar label (suitable for UI status bars).
    pub fn status_bar_label(&self) -> String {
        let e = self.entropy();
        let level = self.level(e);
        let pct = (e * 100.0).round() as u64;
        let bar = self.entropy_bar_chars();
        match level {
            EntropyLevel::Critical => format!("▶ VERIFY │ {pct}% ▓▓▓▓ {bar}"),
            EntropyLevel::High => format!("▶ Verify  │ {pct}% ▓▓▓░ {bar}"),
            EntropyLevel::Medium => format!("  verify  │ {pct}% ▓▓░░ {bar}"),
            EntropyLevel::Low => format!("  verify  │ {pct}% ▓░░░ {bar}"),
        }
    }

    /// Generate a 5-character text bar representing current entropy.
    fn entropy_bar_chars(&self) -> String {
        let entropy = self.entropy();
        let filled = ((entropy * 5.0).round() as usize).min(5);
        let bar: String = std::iter::repeat('▓')
            .take(filled)
            .chain(std::iter::repeat('░').take(5 - filled))
            .collect();
        bar
    }
}

impl Default for VerificationEntropy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Fresh state ──────────────────────────────────────────────────

    #[test]
    fn fresh_tracker_zero_entropy() {
        let ve = VerificationEntropy::new();
        assert_eq!(ve.entropy(), 0.0);
        assert_eq!(ve.current_level(), EntropyLevel::Low);
    }

    #[test]
    fn fresh_tracker_no_edits_no_tests() {
        let ve = VerificationEntropy::new();
        assert_eq!(ve.edits_since_last_test(), 0);
        assert_eq!(ve.total_lines_edited(), 0);
        assert_eq!(ve.total_tests_run(), 0);
    }

    // ── Edit → entropy increases ────────────────────────────────────

    #[test]
    fn edit_increases_entropy() {
        let mut ve = VerificationEntropy::new();
        let e0 = ve.entropy();
        ve.record_edit(10);
        let e1 = ve.entropy();
        assert!(e1 > e0, "edits should increase entropy: {e1} <= {e0}");
    }

    #[test]
    fn edit_increases_since_last_test() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(42);
        assert_eq!(ve.edits_since_last_test(), 42);
    }

    // ── Test → entropy resets ────────────────────────────────────────

    #[test]
    fn test_resets_entropy() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(500);
        assert!(ve.entropy() > 0.3);
        let event = ve.record_test();
        assert_eq!(event.level, EntropyLevel::Low);
        assert_eq!(ve.entropy(), 0.0);
    }

    #[test]
    fn test_resets_edits_since_last_test() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(100);
        assert_eq!(ve.edits_since_last_test(), 100);
        ve.record_test();
        assert_eq!(ve.edits_since_last_test(), 0);
    }

    // ── Bounds ────────────────────────────────────────────────────────

    #[test]
    fn entropy_is_bounded_01() {
        let mut ve = VerificationEntropy::new();
        for _ in 0..100 {
            ve.record_edit(50);
        }
        let e = ve.entropy();
        assert!(
            (0.0..=1.0).contains(&e),
            "entropy should be clamped to [0,1]: got {e}"
        );
    }

    #[test]
    fn entropy_never_exceeds_one() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(10_000);
        let e = ve.entropy();
        assert!(e <= 1.0, "entropy should not exceed 1.0: got {e}");
    }

    #[test]
    fn entropy_never_negative() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(10);
        // Reset to ensure no chance of negative from clamping
        ve.record_test();
        let e = ve.entropy();
        assert!(e >= 0.0, "entropy should not be negative: got {e}");
    }

    // ── Level transitions ─────────────────────────────────────────────

    #[test]
    fn level_transitions_low_to_critical() {
        let mut ve = VerificationEntropy::new();
        assert_eq!(ve.current_level(), EntropyLevel::Low);

        // ~250 lines → should leave Low
        ve.record_edit(250);
        assert!(
            ve.current_level() != EntropyLevel::Low,
            "250 lines should move past Low: got {:?}",
            ve.current_level()
        );

        // +400 (total ~650) → should be High
        ve.record_edit(400);
        assert_eq!(
            ve.current_level(),
            EntropyLevel::High,
            "650 lines should be High: got {:?}",
            ve.current_level()
        );

        // +400 (total ~1050) → should be Critical
        ve.record_edit(400);
        assert_eq!(
            ve.current_level(),
            EntropyLevel::Critical,
            "1050 lines should be Critical: got {:?}",
            ve.current_level()
        );
    }

    #[test]
    fn test_brings_level_back_to_low() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(1000);
        assert_eq!(ve.current_level(), EntropyLevel::Critical);
        ve.record_test();
        assert_eq!(ve.current_level(), EntropyLevel::Low);
    }

    // ── Event generation ──────────────────────────────────────────────

    #[test]
    fn record_edit_triggers_event_at_high() {
        let mut ve = VerificationEntropy::new();
        let mut triggered = false;
        for _ in 0..700 {
            if ve.record_edit(1).is_some() {
                triggered = true;
                break;
            }
        }
        assert!(triggered, "should emit event at High entropy");
    }

    #[test]
    fn record_edit_does_not_trigger_at_low() {
        let mut ve = VerificationEntropy::new();
        for _ in 0..10 {
            assert!(ve.record_edit(1).is_none(), "should not emit at Low");
        }
    }

    #[test]
    fn record_test_returns_discharge_message() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(200);
        let event = ve.record_test();
        assert!(
            event.message.contains("discharged") || event.message.contains("reset"),
            "message: {}",
            event.message
        );
        assert_eq!(event.edits_since_last_test, 0);
    }

    #[test]
    fn record_edits_batch_produces_events() {
        let mut ve = VerificationEntropy::new();
        let events = ve.record_edits(600);
        let has_warning = events
            .iter()
            .any(|e| e.level == EntropyLevel::High || e.level == EntropyLevel::Critical);
        assert!(has_warning, "600 edits should trigger warnings");
    }

    // ── Counters ──────────────────────────────────────────────────────

    #[test]
    fn total_counters_accumulate() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(50);
        ve.record_test();
        ve.record_edit(30);
        ve.record_test();
        ve.record_test();
        assert_eq!(ve.total_tests_run(), 3);
        assert_eq!(ve.total_lines_edited(), 80);
    }

    #[test]
    fn edit_test_ratio_computed() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(100);
        ve.record_test();
        ve.record_edit(50);
        assert_eq!(ve.total_lines_edited(), 150);
        assert_eq!(ve.total_tests_run(), 1);
        assert!((ve.edit_test_ratio() - 150.0).abs() < 0.01);
    }

    #[test]
    fn edit_test_ratio_infinite_when_no_tests() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(10);
        assert_eq!(ve.edit_test_ratio(), f64::MAX);
    }

    // ── Display / formatting ──────────────────────────────────────────

    #[test]
    fn display_format() {
        assert_eq!(format!("{}", EntropyLevel::Low), "low");
        assert_eq!(format!("{}", EntropyLevel::Medium), "medium");
        assert_eq!(format!("{}", EntropyLevel::High), "high");
        assert_eq!(format!("{}", EntropyLevel::Critical), "critical");
    }

    #[test]
    fn status_bar_label_includes_percentage() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(50);
        let label = ve.status_bar_label();
        assert!(!label.is_empty());
        assert!(label.contains('%'));
    }

    #[test]
    fn entropy_bar_chars_at_zero_is_all_empty() {
        let ve = VerificationEntropy::new();
        let bar = ve.entropy_bar_chars();
        assert_eq!(bar.chars().count(), 5);
        assert_eq!(bar, "░░░░░");
    }

    #[test]
    fn build_message_formats_correctly() {
        let mut ve = VerificationEntropy::new();
        let msg = ve.build_message();
        assert!(msg.contains("Good"));
        assert!(msg.contains("0"));

        ve.record_edit(1000);
        let msg = ve.build_message();
        assert!(msg.contains("CONSERVATION"));
        assert!(msg.contains("1000"));
    }

    // ── Custom parameters ─────────────────────────────────────────────

    #[test]
    fn custom_params_affect_entropy_growth() {
        let mut fast = VerificationEntropy::with_params(0.01, 3.0, 0.3, 0.6, 0.8);
        fast.record_edit(100);
        let e_fast = fast.entropy();

        let mut slow = VerificationEntropy::with_params(0.001, 3.0, 0.3, 0.6, 0.8);
        slow.record_edit(100);
        let e_slow = slow.entropy();

        assert!(
            e_fast > e_slow,
            "higher alpha should grown entropy faster: {e_fast} vs {e_slow}"
        );
    }

    #[test]
    fn custom_thresholds_change_level_classification() {
        let mut ve = VerificationEntropy::with_params(0.005, 3.0, 0.10, 0.20, 0.40);
        ve.record_edit(100);
        // With lowered thresholds (medium=0.10), 100 edits should be at least Medium
        let lvl = ve.current_level();
        assert!(
            lvl != EntropyLevel::Low,
            "with low thresholds (medium=0.10), 100 edits should not be Low: got {:?}",
            lvl
        );
        // With even lower thresholds, verify we can reach High
        let mut ve2 = VerificationEntropy::with_params(0.01, 3.0, 0.05, 0.10, 0.20);
        ve2.record_edit(100);
        let lvl2 = ve2.current_level();
        assert!(
            lvl2 == EntropyLevel::High || lvl2 == EntropyLevel::Critical,
            "with aggro thresholds, 100 edits should be high/critical: got {:?}",
            lvl2
        );
    }

    // ── Serialization ─────────────────────────────────────────────────

    #[test]
    fn serialize_deserialize_round_trip() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(200);
        ve.record_test();
        ve.record_edit(75);

        let json = serde_json::to_string(&ve).expect("serialize");
        let deser: VerificationEntropy = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(ve.entropy(), deser.entropy());
        assert_eq!(ve.total_lines_edited(), deser.total_lines_edited());
        assert_eq!(ve.total_tests_run(), deser.total_tests_run());
        assert_eq!(ve.edits_since_last_test(), deser.edits_since_last_test());
        assert_eq!(ve.current_level(), deser.current_level());
    }

    #[test]
    fn serialize_persistence_restores_state() {
        let mut ve = VerificationEntropy::new();
        ve.record_edit(500);
        ve.record_test();
        ve.record_edit(123);

        let json = serde_json::to_string_pretty(&ve).expect("serialize");

        // Simulate saving and loading across sessions
        let restored: VerificationEntropy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.total_lines_edited(), 623);
        assert_eq!(restored.total_tests_run(), 1);
        assert_eq!(restored.edits_since_last_test(), 123);
        assert_eq!(restored.entropy(), ve.entropy());
    }
}
