//! WebAssembly simulation visualization.
//!
//! Compiles to WASM and runs multi-agent simulations in the browser with
//! a Canvas-based visualization. Users can:
//! - Load FST rules (as JSON)
//! - Choose topology and number of agents
//! - Watch the simulation run in real-time
//! - Inject faults (remove agents) and observe self-healing

use agent_model::fst::Fst;
use agent_model::interpreter::{self, AgentState};
use agent_model::topology::Topology;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// The simulation state, held in WASM memory.
#[wasm_bindgen]
pub struct Simulation {
    fst: Fst,
    agents: Vec<AgentState>,
    topology: Topology,
    step: u64,
    history: Vec<Vec<i64>>,
    width: f64,
    height: f64,
}

/// Agent position for rendering (computed from topology + state).
#[derive(Serialize)]
struct AgentRender {
    x: f64,
    y: f64,
    value: i64,
    id: usize,
}

#[wasm_bindgen]
impl Simulation {
    /// Create a new simulation from an FST JSON string.
    #[wasm_bindgen(constructor)]
    pub fn new(
        fst_json: &str,
        num_agents: usize,
        topology_name: &str,
        width: f64,
        height: f64,
    ) -> Result<Simulation, JsValue> {
        let fst: FstJson = serde_json::from_str(fst_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid FST JSON: {}", e)))?;

        let mut fst_model = Fst::new(
            &fst.name,
            fst.num_states,
            fst.num_observations,
            fst.num_actions,
        );

        for (key, (next_state, action)) in &fst.transitions {
            let parts: Vec<&str> = key.split(',').collect();
            if parts.len() == 2 {
                if let (Ok(s), Ok(o)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                    fst_model.add_transition(s, o, *next_state, *action);
                }
            }
        }
        for (key, value) in &fst.output {
            if let Ok(s) = key.parse::<u32>() {
                fst_model.set_output(s, *value);
            }
        }

        let topology = match topology_name {
            "ring" => Topology::Ring,
            "star" => Topology::Star,
            "grid" => {
                let side = (num_agents as f64).sqrt().ceil() as usize;
                Topology::Grid {
                    rows: side,
                    cols: side,
                }
            }
            _ => Topology::Complete,
        };

        // Initialize agents with random-ish values
        let agents: Vec<AgentState> = (0..num_agents)
            .map(|id| {
                let value = (id % fst.num_states as usize) as i64;
                AgentState {
                    id,
                    fst_state: value as u32,
                    value,
                }
            })
            .collect();

        let history = vec![agents.iter().map(|a| a.value).collect()];

        Ok(Simulation {
            fst: fst_model,
            agents,
            topology,
            step: 0,
            history,
            width,
            height,
        })
    }

    /// Advance the simulation by one step. Returns the current step number.
    pub fn step(&mut self) -> u64 {
        let n = self.agents.len();
        let mut new_agents = self.agents.clone();

        for i in 0..n {
            let obs = interpreter::compute_observation(
                i,
                &self.agents,
                &self.topology,
                self.fst.num_observations,
            );
            if let Some((next_state, _action)) = self.fst.step(self.agents[i].fst_state, obs) {
                new_agents[i].fst_state = next_state;
                new_agents[i].value = self.fst.get_output(next_state).unwrap_or(next_state as i64);
            }
        }

        self.agents = new_agents;
        self.step += 1;
        self.history
            .push(self.agents.iter().map(|a| a.value).collect());
        self.step
    }

    /// Remove an agent (simulate failure).
    pub fn remove_agent(&mut self, id: usize) -> bool {
        if let Some(pos) = self.agents.iter().position(|a| a.id == id) {
            self.agents.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if all agents agree on the same value.
    pub fn all_agree(&self) -> bool {
        if self.agents.is_empty() {
            return true;
        }
        let first = self.agents[0].value;
        self.agents.iter().all(|a| a.value == first)
    }

    /// Get the current step number.
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /// Get the number of agents.
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Get the agent states as a JSON string for rendering.
    pub fn agents_json(&self) -> String {
        let n = self.agents.len();
        let renders: Vec<AgentRender> = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let (x, y) = self.agent_position(i, n);
                AgentRender {
                    x,
                    y,
                    value: a.value,
                    id: a.id,
                }
            })
            .collect();
        serde_json::to_string(&renders).unwrap_or_default()
    }

    /// Get the value history as JSON for charting.
    pub fn history_json(&self) -> String {
        serde_json::to_string(&self.history).unwrap_or_default()
    }

    /// Render the simulation to a canvas context.
    pub fn render(&self, ctx: &web_sys::CanvasRenderingContext2d) {
        let n = self.agents.len();

        // Clear canvas
        ctx.set_fill_style_str("#1a1a2e");
        ctx.fill_rect(0.0, 0.0, self.width, self.height);

        // Draw edges
        ctx.set_stroke_style_str("#333355");
        ctx.set_line_width(0.5);
        for i in 0..n {
            let neighbors = self.topology.neighbors(self.agents[i].id, n);
            let (x1, y1) = self.agent_position(i, n);
            for &j_id in &neighbors {
                if let Some(j) = self.agents.iter().position(|a| a.id == j_id) {
                    let (x2, y2) = self.agent_position(j, n);
                    ctx.begin_path();
                    ctx.move_to(x1, y1);
                    ctx.line_to(x2, y2);
                    ctx.stroke();
                }
            }
        }

        // Draw agents
        let radius = 8.0_f64.max(200.0 / n as f64);
        for (i, agent) in self.agents.iter().enumerate() {
            let (x, y) = self.agent_position(i, n);

            // Color based on value
            let color = match agent.value {
                0 => "#e94560",
                1 => "#0f3460",
                2 => "#16c79a",
                3 => "#f5a623",
                _ => "#888888",
            };

            ctx.set_fill_style_str(color);
            ctx.begin_path();
            let _ = ctx.arc(x, y, radius, 0.0, std::f64::consts::TAU);
            ctx.fill();

            // Label
            ctx.set_fill_style_str("#ffffff");
            ctx.set_font(&format!("{}px monospace", (radius * 0.8).max(10.0)));
            ctx.set_text_align("center");
            ctx.set_text_baseline("middle");
            let _ = ctx.fill_text(&agent.value.to_string(), x, y);
        }

        // Status text
        ctx.set_fill_style_str("#ffffff");
        ctx.set_font("14px monospace");
        ctx.set_text_align("left");
        ctx.set_text_baseline("top");
        let status = format!(
            "Step: {}  Agents: {}  Consensus: {}",
            self.step,
            n,
            if self.all_agree() { "YES" } else { "NO" }
        );
        let _ = ctx.fill_text(&status, 10.0, 10.0);
    }
}

impl Simulation {
    /// Compute the position of agent i for rendering.
    fn agent_position(&self, i: usize, n: usize) -> (f64, f64) {
        let cx = self.width / 2.0;
        let cy = self.height / 2.0;
        let r = (self.width.min(self.height) / 2.0) * 0.8;

        match &self.topology {
            Topology::Ring | Topology::Complete | Topology::KNearest { .. } => {
                // Circular layout
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64
                    - std::f64::consts::FRAC_PI_2;
                (cx + r * angle.cos(), cy + r * angle.sin())
            }
            Topology::Grid { cols, .. } => {
                let col = i % cols;
                let row = i / cols;
                let spacing_x = self.width / (*cols as f64 + 1.0);
                let spacing_y = self.height / ((n / cols + 1) as f64 + 1.0);
                (spacing_x * (col as f64 + 1.0), spacing_y * (row as f64 + 1.0))
            }
            Topology::Star => {
                if i == 0 {
                    (cx, cy)
                } else {
                    let angle = 2.0 * std::f64::consts::PI * (i - 1) as f64 / (n - 1).max(1) as f64
                        - std::f64::consts::FRAC_PI_2;
                    (cx + r * angle.cos(), cy + r * angle.sin())
                }
            }
            Topology::RandomGeometric { .. } => {
                // Use agent id to compute a deterministic position
                let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
                let theta = 2.0 * std::f64::consts::PI * (i as f64 * golden).fract();
                let radius = r * ((i as f64 + 1.0) / n as f64).sqrt();
                (cx + radius * theta.cos(), cy + radius * theta.sin())
            }
        }
    }
}

/// JSON-serializable FST format (matches what the CLI outputs).
#[derive(Deserialize)]
struct FstJson {
    name: String,
    num_states: u32,
    num_observations: u32,
    num_actions: u32,
    transitions: std::collections::HashMap<String, (u32, u32)>,
    output: std::collections::HashMap<String, i64>,
}
