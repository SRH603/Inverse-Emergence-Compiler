//! Finite State Transducer (FST) — the simplest agent rule representation.
//!
//! An FST maps (current_state, observation) → (next_state, action).
//! All agents share the same FST (symmetric/uniform rules).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A state in the FST, represented as an index.
pub type StateId = u32;

/// An observation symbol — what the agent sees about its neighbors.
/// For consensus: the majority value among neighbors.
/// Abstracted as a finite alphabet.
pub type ObservationId = u32;

/// An action symbol — what the agent does.
pub type ActionId = u32;

/// A Finite State Transducer defining an agent's local rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fst {
    /// Human-readable name
    pub name: String,
    /// Number of internal states
    pub num_states: u32,
    /// Number of possible observations
    pub num_observations: u32,
    /// Number of possible actions
    pub num_actions: u32,
    /// Initial state
    pub initial_state: StateId,
    /// Transition function: (state, observation) → (next_state, action)
    pub transitions: HashMap<(StateId, ObservationId), (StateId, ActionId)>,
    /// Output value function: state → value (for consensus-like problems)
    pub output: HashMap<StateId, i64>,
}

impl Fst {
    /// Create a new empty FST with the given dimensions.
    pub fn new(
        name: impl Into<String>,
        num_states: u32,
        num_observations: u32,
        num_actions: u32,
    ) -> Self {
        Self {
            name: name.into(),
            num_states,
            num_observations,
            num_actions,
            initial_state: 0,
            transitions: HashMap::new(),
            output: HashMap::new(),
        }
    }

    /// Add a transition.
    pub fn add_transition(
        &mut self,
        from: StateId,
        obs: ObservationId,
        to: StateId,
        action: ActionId,
    ) {
        self.transitions.insert((from, obs), (to, action));
    }

    /// Set the output value for a state.
    pub fn set_output(&mut self, state: StateId, value: i64) {
        self.output.insert(state, value);
    }

    /// Step the FST: given current state and observation, return (next_state, action).
    pub fn step(&self, state: StateId, obs: ObservationId) -> Option<(StateId, ActionId)> {
        self.transitions.get(&(state, obs)).copied()
    }

    /// Get the output value of a state.
    pub fn get_output(&self, state: StateId) -> Option<i64> {
        self.output.get(&state).copied()
    }

    /// Total number of possible transitions (for enumeration bounds).
    pub fn transition_space_size(&self) -> u64 {
        let entries = self.num_states as u64 * self.num_observations as u64;
        let choices = self.num_states as u64 * self.num_actions as u64;
        choices.pow(entries as u32) // This is huge even for small FSTs
    }

    /// Check if the FST is complete (all transitions defined).
    pub fn is_complete(&self) -> bool {
        for s in 0..self.num_states {
            for o in 0..self.num_observations {
                if !self.transitions.contains_key(&(s, o)) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if this FST is degenerate (trivial/useless).
    ///
    /// A degenerate FST is one that:
    /// - Ignores observations entirely (all observations lead to the same next state)
    /// - Has only one reachable state
    /// - All states produce the same output value
    /// - Never changes state regardless of input
    pub fn is_degenerate(&self) -> bool {
        if self.num_states <= 1 {
            return true;
        }

        // Check: does the FST ignore observations?
        // For each state, check if all observations lead to the same (next_state, action)
        let ignores_obs = (0..self.num_states).all(|s| {
            let first = self.transitions.get(&(s, 0));
            (1..self.num_observations).all(|o| self.transitions.get(&(s, o)) == first)
        });

        if ignores_obs {
            return true;
        }

        // Check: do all states have the same output value?
        if !self.output.is_empty() {
            let first_val = self.output.values().next();
            if let Some(first) = first_val {
                if self.output.values().all(|v| v == first) {
                    return true;
                }
            }
        }

        // Check: is only one state reachable from initial_state?
        let mut reachable = std::collections::HashSet::new();
        let mut frontier = vec![self.initial_state];
        while let Some(s) = frontier.pop() {
            if !reachable.insert(s) {
                continue;
            }
            for o in 0..self.num_observations {
                if let Some((next, _)) = self.transitions.get(&(s, o)) {
                    if !reachable.contains(next) {
                        frontier.push(*next);
                    }
                }
            }
        }
        if reachable.len() <= 1 {
            return true;
        }

        // Check: do all reachable states produce the same output?
        let reachable_outputs: std::collections::HashSet<i64> = reachable
            .iter()
            .filter_map(|s| self.output.get(s))
            .copied()
            .collect();
        if reachable_outputs.len() <= 1 {
            return true;
        }

        false
    }

    /// Check if the FST preserves initial diversity: different initial states
    /// can lead to different consensus values, not always the same one.
    pub fn preserves_majority(&self) -> bool {
        // Check that at least two different observations lead to different states
        let mut obs_to_next: std::collections::HashSet<StateId> = std::collections::HashSet::new();
        for o in 0..self.num_observations {
            if let Some((next, _)) = self.transitions.get(&(self.initial_state, o)) {
                obs_to_next.insert(*next);
            }
        }
        obs_to_next.len() > 1
    }
}

/// Enumerate all possible complete FSTs with given dimensions.
/// WARNING: This is exponential. Only use for tiny FSTs (2-3 states, 2-3 observations).
pub struct FstEnumerator {
    num_states: u32,
    num_observations: u32,
    num_actions: u32,
    /// Current assignment: for each (state, obs) pair, which (next_state, action) index
    current: Vec<u64>,
    /// Total choices per transition entry
    choices_per_entry: u64,
    /// Whether we've exhausted all possibilities
    done: bool,
}

impl FstEnumerator {
    pub fn new(num_states: u32, num_observations: u32, num_actions: u32) -> Self {
        let entries = (num_states * num_observations) as usize;
        let choices = num_states as u64 * num_actions as u64;
        Self {
            num_states,
            num_observations,
            num_actions,
            current: vec![0; entries],
            choices_per_entry: choices,
            done: entries == 0,
        }
    }

    /// Get the current FST, or None if exhausted.
    pub fn current_fst(&self) -> Option<Fst> {
        if self.done {
            return None;
        }

        let mut fst = Fst::new(
            "enumerated",
            self.num_states,
            self.num_observations,
            self.num_actions,
        );

        let mut idx = 0;
        for s in 0..self.num_states {
            for o in 0..self.num_observations {
                let choice = self.current[idx];
                let next_state = (choice / self.num_actions as u64) as u32;
                let action = (choice % self.num_actions as u64) as u32;
                fst.add_transition(s, o, next_state, action);
                idx += 1;
            }
        }

        // Default output: state id as value
        for s in 0..self.num_states {
            fst.set_output(s, s as i64);
        }

        Some(fst)
    }

    /// Advance to the next FST. Returns false if exhausted.
    pub fn advance(&mut self) -> bool {
        if self.done {
            return false;
        }

        // Increment like a mixed-radix counter
        for i in 0..self.current.len() {
            self.current[i] += 1;
            if self.current[i] < self.choices_per_entry {
                return true;
            }
            self.current[i] = 0;
        }

        self.done = true;
        false
    }

    /// Total number of FSTs to enumerate.
    pub fn total_count(&self) -> u64 {
        let entries = self.num_states as u64 * self.num_observations as u64;
        if entries == 0 {
            return 0;
        }
        self.choices_per_entry.saturating_pow(entries as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fst_step() {
        let mut fst = Fst::new("test", 2, 2, 2);
        fst.add_transition(0, 0, 0, 0);
        fst.add_transition(0, 1, 1, 1);
        fst.add_transition(1, 0, 0, 0);
        fst.add_transition(1, 1, 1, 1);
        fst.set_output(0, 0);
        fst.set_output(1, 1);

        assert_eq!(fst.step(0, 0), Some((0, 0)));
        assert_eq!(fst.step(0, 1), Some((1, 1)));
        assert!(fst.is_complete());
    }

    #[test]
    fn test_enumerator_count() {
        // 2 states, 2 observations, 1 action
        // Entries: 2*2 = 4, choices per entry: 2*1 = 2
        // Total: 2^4 = 16
        let e = FstEnumerator::new(2, 2, 1);
        assert_eq!(e.total_count(), 16);
    }

    #[test]
    fn test_enumerator_iterate() {
        let mut e = FstEnumerator::new(2, 2, 1);
        let mut count = 0;
        loop {
            if e.current_fst().is_none() {
                break;
            }
            count += 1;
            if !e.advance() {
                break;
            }
        }
        // Should see all 16 FSTs (the last one before advance returns false)
        assert_eq!(count, 16);
    }

    #[test]
    fn test_degenerate_single_state() {
        let mut fst = Fst::new("single", 1, 2, 1);
        fst.add_transition(0, 0, 0, 0);
        fst.add_transition(0, 1, 0, 0);
        fst.set_output(0, 0);
        assert!(fst.is_degenerate());
    }

    #[test]
    fn test_degenerate_ignores_obs() {
        let mut fst = Fst::new("ignores", 2, 2, 1);
        // Both observations lead to same next state
        fst.add_transition(0, 0, 1, 0);
        fst.add_transition(0, 1, 1, 0);
        fst.add_transition(1, 0, 0, 0);
        fst.add_transition(1, 1, 0, 0);
        fst.set_output(0, 0);
        fst.set_output(1, 1);
        assert!(fst.is_degenerate());
    }

    #[test]
    fn test_degenerate_same_output() {
        let mut fst = Fst::new("same_out", 2, 2, 1);
        fst.add_transition(0, 0, 0, 0);
        fst.add_transition(0, 1, 1, 0);
        fst.add_transition(1, 0, 0, 0);
        fst.add_transition(1, 1, 1, 0);
        fst.set_output(0, 42);
        fst.set_output(1, 42); // same output for all states
        assert!(fst.is_degenerate());
    }

    #[test]
    fn test_non_degenerate() {
        let mut fst = Fst::new("good", 2, 2, 1);
        fst.add_transition(0, 0, 0, 0);
        fst.add_transition(0, 1, 1, 0);
        fst.add_transition(1, 0, 0, 0);
        fst.add_transition(1, 1, 1, 0);
        fst.set_output(0, 0);
        fst.set_output(1, 1);
        assert!(!fst.is_degenerate());
    }

    #[test]
    fn test_preserves_majority() {
        let mut fst = Fst::new("good", 2, 2, 1);
        fst.add_transition(0, 0, 0, 0);
        fst.add_transition(0, 1, 1, 0);
        fst.set_output(0, 0);
        fst.set_output(1, 1);
        assert!(fst.preserves_majority());
    }
}
