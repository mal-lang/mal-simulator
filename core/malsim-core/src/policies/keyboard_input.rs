//! Port of `keyboard_input.py`.

use std::io::{self, BufRead, Write};

use maltoolbox_attackgraph::AttackGraphNodeId;

use crate::agent_state::AgentState;
use crate::policies::{DecisionAgent, PolicyError};

/// An agent that makes decisions by asking user for keyboard input.
///
/// Generic over its input/output so it can be driven by something other
/// than a terminal; `KeyboardAgent::new()` uses stdin/stdout.
pub struct KeyboardAgent<R = io::BufReader<io::Stdin>, W = io::Stdout> {
    input: R,
    output: W,
}

impl KeyboardAgent {
    pub fn new() -> Self {
        KeyboardAgent {
            input: io::BufReader::new(io::stdin()),
            output: io::stdout(),
        }
    }
}

impl Default for KeyboardAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: BufRead, W: Write> KeyboardAgent<R, W> {
    pub fn with_io(input: R, output: W) -> Self {
        KeyboardAgent { input, output }
    }
}

/// What the user typed: `None` for an invalid entry, `Some(None)` for an
/// empty line (wait), `Some(Some(index))` for a listed action.
fn parse_action(user_input: &str, num_actions: usize) -> Option<Option<usize>> {
    if user_input.is_empty() {
        return Some(None);
    }
    match user_input.parse::<usize>() {
        Ok(index) if index < num_actions => Some(Some(index)),
        _ => None,
    }
}

impl<R: BufRead, W: Write> DecisionAgent for KeyboardAgent<R, W> {
    /// Compute action from action_surface.
    fn get_next_action(
        &mut self,
        agent_state: &AgentState<'_>,
    ) -> Result<Option<AttackGraphNodeId>, PolicyError> {
        let index_to_node = agent_state.sorted_action_surface();
        if index_to_node.is_empty() {
            writeln!(self.output, "No actions to pick for defender")?;
            return Ok(None);
        }
        let full_name = |node| {
            agent_state
                .attack_graph()
                .full_name_of(node, Some(agent_state.model()))
        };

        let index = loop {
            writeln!(self.output, "Available actions:")?;
            for (i, &node) in index_to_node.iter().enumerate() {
                writeln!(self.output, "{i}. {}", full_name(node))?;
            }
            writeln!(self.output, "Enter action or leave empty to wait:")?;
            write!(self.output, "> ")?;
            self.output.flush()?;

            let mut user_input = String::new();
            if self.input.read_line(&mut user_input)? == 0 {
                // Python's `input()` raises `EOFError` here.
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
            }
            match parse_action(user_input.trim(), index_to_node.len()) {
                Some(index) => break index,
                None => writeln!(self.output, "Invalid action.")?,
            }
        };

        match index {
            Some(index) => {
                let node = index_to_node[index];
                writeln!(self.output, "Selected action: {}", full_name(node))?;
                Ok(Some(node))
            }
            None => {
                writeln!(self.output, "wait")?;
                Ok(None)
            }
        }
    }
}
