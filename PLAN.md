# Inverse Emergence Compiler — Full Implementation Plan

## Literature Positioning

Based on comprehensive survey of 60+ papers (2015-2025), the key findings:

**What exists:**
- **Distributed synthesis** (Finkbeiner/Schewe bounded synthesis, Jacobs/Bloem parameterized synthesis): synthesizes distributed protocols from LTL specs, but limited to token-ring topologies, no emergence reasoning
- **Protocol verification** (IronFleet, Verdi, Grove, DistAI, DuoAI, Kondo, Basilisk): verifies hand-written protocols, does NOT synthesize from scratch. Basilisk (OSDI 2025 Best Paper) auto-generates provenance invariants but still requires human-designed protocols
- **Cinnabar** (CAV 2023): closest to our goal — CEGIS-based synthesis of distributed agreement protocols, but requires human-provided sketches
- **Neural Cellular Automata** (Mordvintsev 2020, DiffLogic CA 2025): learns local CA rules via gradient descent to produce target patterns, but rules are neural/opaque; DiffLogic CA extracts discrete rules but limited to grid CAs
- **Graph Neural Cellular Automata** (Grattarola, NeurIPS 2021): GNN-based local rules on arbitrary graphs, proven to represent any GCA, but only learns to imitate existing dynamics, not synthesize for specs
- **SwarmSTL/GTL** (Yan 2019, Djeumou RSS 2020): temporal logics for swarm properties, but used for monitoring/planning, not rule synthesis
- **Causal Emergence** (Hoel 2025 "Engineering Emergence"): shows emergence can be engineered with precision on discrete systems — directly relevant theoretical foundation
- **Mean-Field Games** (Graphon MFG, Caines 2021): bridges individual-to-collective via PDE limits, inverse MFG recovers costs from behavior — relevant mathematical framework

**What does NOT exist (our gap):**
- No system takes a formal spec of desired emergent behavior and synthesizes local agent rules
- No programming language for specifying emergent properties (as opposed to individual agent behavior)
- No formal verification framework specifically for emergence (verifying that local rules produce global emergent behavior)
- The intersection of {program synthesis} × {emergence} × {formal verification} is completely empty

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                   EmergeLang                     │
│         (Specification Language Layer)            │
│  Global invariants, convergence, fault-tolerance │
└──────────────────────┬──────────────────────────┘
                       │ AST + Formal Spec (LTL/CTL extension)
                       ▼
┌─────────────────────────────────────────────────┐
│              Rule Synthesizer                    │
│  ┌───────────────┐  ┌────────────────────────┐  │
│  │ Symbolic Path │  │ Neural-Guided Path     │  │
│  │ (SMT/CEGIS)  │  │ (GNN + DiffLogic)      │  │
│  └───────┬───────┘  └──────────┬─────────────┘  │
│          └──────────┬──────────┘                 │
│                     ▼                            │
│          Candidate Local Rules (FST/Programs)    │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│            Emergence Verifier                    │
│  ┌────────────────┐  ┌───────────────────────┐  │
│  │ Parameterized  │  │ Lyapunov/Abstract     │  │
│  │ Model Checking │  │ Interpretation        │  │
│  └────────────────┘  └───────────────────────┘  │
│  ┌────────────────────────────────────────────┐  │
│  │ Statistical Verification (Monte Carlo)     │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────┘
                       │ Verified Local Rules
                       ▼
┌─────────────────────────────────────────────────┐
│              Swarm Runtime                       │
│  Code generation for physical/virtual agents     │
└─────────────────────────────────────────────────┘
```

---

## Implementation Language Choice: Rust

Rust for the compiler core (parser, synthesizer, verifier). Reasons:
- Performance-critical synthesis loops (SMT solver integration, parallel simulation)
- Memory safety without GC (important for the runtime targeting embedded/IoT)
- Strong type system aligns with PL research culture
- Excellent FFI for integrating Z3 (C API), Python (PyO3 for neural components)
- The Swarm Runtime can compile to WebAssembly for simulation or native for deployment

Python for the neural-guided synthesis path (PyTorch/JAX for GNN training).

---

## Phase 1: EmergeLang — The Specification Language

### 1.1 Core Language Design

EmergeLang is NOT a general-purpose programming language. It is a **specification language** for describing emergent properties. The key insight: we need to express properties about **collections of agents** and their **collective dynamics**, not about individual computations.

#### Ontological Primitives

```
// Agent types with local state schemas
agent_type Drone {
    state: { position: Vec3, velocity: Vec3, battery: Float }
    observe: neighbors_within(radius: Float) -> Set<AgentView>
    act: { move(Vec3), broadcast(Message) }
}

// Topology specifications
topology Grid(rows: Nat, cols: Nat)
topology KNearestGraph(k: Nat)
topology DynamicRadius(r: Float)

// Emergent property specifications using temporal + spatial logic
emerge Coverage(drones: Swarm<Drone>, area: Region) {
    // Eventually, the union of sensed areas covers the target
    eventually globally: union(d.sense_area for d in drones) ⊇ area

    // No single drone is critical (fault tolerance)
    forall d in drones:
        without(d): eventually globally: union(d'.sense_area for d' in drones \ {d}) ⊇ area

    // Convergence bound
    converge_within: 100 * |drones| steps
}
```

#### Property Classes (what EmergeLang can express)

1. **Spatial properties**: coverage, partitioning, formation, connectivity
2. **Temporal properties**: convergence, stability, oscillation, phase transitions
3. **Fault tolerance**: k-resilience, graceful degradation, self-healing
4. **Statistical properties**: distribution convergence, entropy bounds
5. **Graph properties**: small-world, scale-free, community structure
6. **Game-theoretic properties**: Nash equilibrium, Pareto optimality, mechanism properties

#### Formal Semantics

EmergeLang compiles to an extended temporal logic we call **Emergence Temporal Logic (ETL)**:
- Base: CTL* (branching-time temporal logic, subsumes both LTL and CTL)
- Extension 1: **Aggregate quantifiers** — `fraction(P, agents) >= 0.9` means "at least 90% of agents satisfy P"
- Extension 2: **Topological modalities** — `nearby(d, r) |= P` means "all agents within radius r of d satisfy P"
- Extension 3: **Convergence operators** — `converges_to(f(agents), target, epsilon, T)` means "the function f evaluated on the agent collective approaches target within epsilon by time T"
- Extension 4: **Perturbation quantifiers** — `robust(P, remove(k))` means "P holds even after removing any k agents"

### 1.2 Parser & Type System

- PEG parser (use `pest` crate in Rust) → AST
- Bidirectional type checking: ensure agent state schemas are consistent with observation/action capabilities
- Semantic analysis: check that emergent properties reference valid agent types, valid state fields, and that quantifiers are well-formed
- Compilation to ETL formulas: desugar high-level specs into core ETL

### 1.3 Deliverables
- `emergelang/` — parser, AST types, type checker, ETL compiler
- Grammar specification (PEG)
- ~10 example specs covering each property class

---

## Phase 2: Agent Model & Rule Representation

### 2.1 What is a "Local Rule"?

A local rule is a function: `(local_state, observations) → (new_state, actions)`

We support three representations at increasing expressiveness (and decreasing verifiability):

**Level 1: Finite State Transducer (FST)**
- Finite local state, finite observation alphabet, deterministic transitions
- Fully verifiable via parameterized model checking
- Sufficient for: consensus, leader election, mutual exclusion, simple formations

**Level 2: Guarded Command Program (GCP)**
- Variables over bounded integer domains
- Guards: boolean conditions on local state + observations
- Actions: variable updates + message sends
- Verifiable via bounded model checking + abstraction
- Sufficient for: threshold protocols, gossip protocols, gradient-based formations

**Level 3: Continuous Program (CP)**
- Real-valued state, continuous observation space
- Update rules are arithmetic expressions (addition, multiplication, min/max, thresholding)
- Verifiable via Lyapunov analysis + statistical testing
- Sufficient for: flocking, consensus in continuous space, coverage control

### 2.2 Rule Space Definition

For each level, define the **rule template** (syntactic structure) and **parameter space**:

```
// Level 2 example: a guarded command template for consensus
rule_template ConsensusRule {
    params: { threshold: Nat, weight: Float }
    variables: { value: Int, decided: Bool }

    on receive(v) from neighbor:
        if count(neighbors where neighbor.value == v) >= params.threshold:
            value := v
            decided := true
}
```

The synthesizer searches over `params` and over the structure of guards/actions.

### 2.3 Deliverables
- `agent_model/` — FST, GCP, CP representations + serialization
- Rule template language and parser
- Rule interpreter (for simulation)

---

## Phase 3: Rule Synthesizer

This is the hardest layer. Two parallel synthesis paths that feed into each other.

### 3.1 Symbolic Synthesis Path (SMT/CEGIS)

**Architecture**: Counterexample-Guided Inductive Synthesis (CEGIS) loop

```
1. Start with rule template T (structural skeleton with parameter holes)
2. Propose candidate parameters θ via SMT solver
3. Model-check the parameterized system (T[θ], N agents) against ETL spec
   - Use cutoff results: if topology has cutoff k, only check k agents
   - For parameterized families: use ByMC-style threshold abstraction
4. If model checker finds counterexample (trace violating spec):
   - Add counterexample as constraint to SMT
   - Go to step 2
5. If no counterexample found within cutoff: candidate passes
6. Forward to Verifier (Phase 4) for full proof
```

**Key technical contributions needed:**

a) **ETL-to-SMT encoding**: Encode the requirement "N agents running rule R satisfy ETL formula φ" as an SMT formula. For finite-state agents, this is a standard bounded model checking encoding. For guarded commands, use bitvector theory. The novelty is encoding *aggregate* properties (fraction quantifiers, convergence operators).

b) **Cutoff computation for ETL**: Extend existing cutoff results (which work for LTL over token rings) to ETL over richer topologies. Specifically:
- For symmetric agents on complete graphs: leverage Emerson-Namjoshi style cutoffs
- For agents on bounded-degree graphs: develop new cutoff results based on neighborhood structure
- For dynamic topologies: use abstract topology classes (any graph with min-degree k)

c) **Template inference**: Before CEGIS, infer promising rule templates from the spec structure. E.g., a coverage spec naturally suggests a repulsion-attraction rule template; a consensus spec suggests a threshold-voting template. Use a library of template families + heuristic matching.

**Tools to integrate**: Z3 (SMT solver), nuXmv (model checker), ByMC (parameterized model checker for threshold protocols)

### 3.2 Neural-Guided Synthesis Path (GNN + Differentiable Logic)

For problems where symbolic synthesis is too slow (large state spaces, continuous domains):

**Architecture**: Inspired by DiffLogic CA (Mordvintsev 2025) + GNCA (Grattarola 2021)

```
1. Define agent interaction as a message-passing GNN:
   - Each agent is a node; edges defined by topology
   - Local rule = GNN update function (shared across all nodes)
   - Rule parameters = GNN weights

2. Define differentiable loss from ETL spec:
   - Coverage → negative of area covered (differentiable approximation)
   - Convergence → distance to target distribution over time
   - Fault tolerance → loss averaged over random agent removals

3. Train GNN via gradient descent to minimize loss
   - Use curriculum learning: start with small N, increase
   - Use random topology augmentation for generalization

4. Extract discrete rules from trained GNN:
   - DiffLogic approach: replace neurons with differentiable logic gates,
     anneal temperature to discretize
   - Decision tree distillation: fit interpretable decision tree to GNN behavior
   - Program synthesis from I/O examples: use GNN behavior as oracle,
     synthesize a GCP that matches it

5. Forward extracted rules to Verifier (Phase 4)
```

**Key technical contributions needed:**

a) **Differentiable ETL loss functions**: Many ETL properties are not naturally differentiable (temporal operators, quantifiers over agents). Need smooth relaxations:
- `eventually P` → `max_t sigmoid(P(t))` with temperature
- `fraction(P, agents) >= c` → `sigmoid(mean(P(agent_i)) - c)`
- `converges_to(f, target, eps, T)` → `||f(agents, T) - target||² + penalty(||f(agents, t) - target|| for t > T)`

b) **GNN architecture for emergence**: Standard GNNs suffer from oversmoothing (all nodes converge to same representation after many layers). Our GNN needs to maintain *diversity* while achieving *coordination*. Explore:
- Positional encodings (random features, Laplacian eigenvectors)
- Heterogeneous message passing (different message types)
- Attention-based neighbor weighting

c) **Rule extraction**: The gap between neural and symbolic. DiffLogic CA showed this is feasible for grid CAs. We need to extend to arbitrary graphs and richer rule languages (GCPs, not just logic gates).

**Tools**: PyTorch Geometric (GNN), JAX (differentiable simulation)

### 3.3 Hybrid Loop

The two paths reinforce each other:
- Neural path finds approximate solutions fast → provides good starting templates for symbolic path
- Symbolic path provides counterexamples → hard training examples for neural path
- Neural path discovers unexpected rule structures → expands template library for symbolic path

### 3.4 Deliverables
- `synthesizer/symbolic/` — CEGIS engine, ETL-to-SMT encoder, cutoff computation
- `synthesizer/neural/` — GNN trainer, differentiable ETL losses, rule extraction
- `synthesizer/hybrid/` — orchestration of both paths
- Integration with Z3, nuXmv

---

## Phase 4: Emergence Verifier

Given candidate local rules R and ETL spec φ, prove that R ⊨ φ for all N ≥ N_min.

### 4.1 Tier 1: Parameterized Model Checking (for FST/GCP rules)

For finite-state agents:
- Compute cutoff k for the given spec + topology class
- Model-check the concrete system with k agents
- If passes: guaranteed for all N ≥ k

Technology: extend ByMC's threshold abstraction to ETL aggregate quantifiers.

### 4.2 Tier 2: Lyapunov-based Convergence Proofs (for CP rules)

For continuous-state agents with convergence specs:

```
1. Hypothesize a Lyapunov function V: GlobalState → ℝ≥0
   - V = 0 iff the emergent property is satisfied
   - Templates: sum of pairwise distances, variance of agent states,
     graph Laplacian quadratic form, entropy functions

2. Prove V is strictly decreasing along system trajectories:
   - Encode V(state(t+1)) < V(state(t)) as SMT formula
   - Use Dreal (δ-decidable SMT over reals) for nonlinear arithmetic

3. If proof succeeds: system converges to emergent property from any initial state

4. If proof fails: use counterexample to refine V (template-based Lyapunov synthesis)
```

Integrate with dReal (δ-complete SMT solver for nonlinear real arithmetic).

### 4.3 Tier 3: Statistical Verification (for all levels)

When formal proof is infeasible:

```
1. Run M independent simulations with random initial conditions
   - Vary N (number of agents), topology, initial states
   - Inject faults (agent removal, message loss) per fault-tolerance spec

2. For each simulation, check if ETL spec is satisfied (runtime monitoring)
   - Use SwarmSTL-style monitoring algorithms

3. Compute statistical guarantee:
   - Bayesian confidence interval on P(spec satisfied)
   - Report: "with 99.9% confidence, the spec is satisfied with probability ≥ 0.99"

4. Identify failure modes:
   - Cluster failing initial conditions
   - Report which aspects of the spec are most fragile
```

### 4.4 Verification Certificate

Output a machine-checkable certificate:
- For Tier 1: the cutoff k + model checking trace
- For Tier 2: the Lyapunov function + SMT proof
- For Tier 3: simulation parameters + statistical report

### 4.5 Deliverables
- `verifier/parameterized/` — cutoff computation, model checking integration
- `verifier/lyapunov/` — Lyapunov template synthesis, dReal integration
- `verifier/statistical/` — parallel simulation engine, runtime monitoring, statistics
- `verifier/certificate/` — proof certificate format and checker

---

## Phase 5: Swarm Runtime

### 5.1 Code Generation Targets

From verified local rules, generate executable code for:

a) **Simulation** (WebAssembly):
- Browser-based visualization of emergent behavior
- Interactive: user can add/remove agents, inject faults, change parameters
- Uses wgpu for GPU-accelerated rendering of large swarms

b) **Embedded/Robotics** (C/Rust no_std):
- Compiles FST/GCP rules to bare-metal C or Rust no_std
- Minimal memory footprint, deterministic execution
- Communication layer abstraction (WiFi, BLE, LoRa)
- Targets: STM32, ESP32, PX4 (drone autopilot)

c) **Distributed Systems** (Rust async):
- Compiles rules to async Rust actors (tokio-based)
- Network communication via gRPC or custom UDP protocol
- Monitoring: runtime ETL checker to detect emergence violations

### 5.2 Runtime Services
- **Topology manager**: maintains neighbor lists as agents move/join/leave
- **Fault detector**: heartbeat-based failure detection
- **Telemetry**: reports agent states to a central dashboard for observation (not control)
- **Hot-reload**: update local rules without restarting agents

### 5.3 Deliverables
- `runtime/wasm/` — WebAssembly simulation target
- `runtime/embedded/` — C/Rust code generator for microcontrollers
- `runtime/distributed/` — async Rust runtime for networked agents
- `runtime/monitor/` — runtime ETL monitoring

---

## Phase 6: Benchmark Suite & Evaluation

### 6.1 Benchmark Problems (increasing difficulty)

| # | Problem | Agent Model | Spec Type | Baseline |
|---|---------|-------------|-----------|----------|
| 1 | Symmetric consensus | FST | Safety + Liveness | Raft, Paxos |
| 2 | 2D coverage | CP | Spatial + Convergence | Voronoi-based (Lloyd's) |
| 3 | Flocking (Reynolds) | CP | Formation + Fault tolerance | Boids |
| 4 | Self-healing ring | GCP | Topology + k-resilience | Chord DHT |
| 5 | Traffic signal coordination | GCP | Flow optimization + Fairness | Webster's formula |
| 6 | Byzantine consensus | FST | Safety + Liveness + Byzantine | PBFT, Tendermint |
| 7 | Multi-species foraging | CP | Task allocation + Efficiency | AutoMoDe |
| 8 | Small-world network formation | GCP | Graph property + Robustness | Watts-Strogatz |

### 6.2 Evaluation Metrics

- **Synthesis time**: wall-clock time from spec to verified rules
- **Rule complexity**: number of states/guards/parameters in synthesized rules
- **Performance vs. hand-designed**: compare emergent behavior quality (convergence speed, fault tolerance, throughput) against known hand-designed solutions
- **Scalability**: how synthesis/verification time scales with N (number of agents) and spec complexity
- **Generalization**: do rules synthesized for N=100 work for N=1000?

### 6.3 Deliverables
- `benchmarks/` — all 8 benchmark problems with EmergeLang specs
- `evaluation/` — automated evaluation pipeline
- Paper-ready comparison tables and plots

---

## Implementation Order & Milestones

### Milestone 0: Project Skeleton (Week 1-2)
- [ ] Rust workspace with crate structure matching architecture
- [ ] CI/CD (GitHub Actions): build, test, lint (clippy), format
- [ ] Python environment for neural components (pyproject.toml, PyO3 bridge)
- [ ] Basic README with architecture diagram

### Milestone 1: EmergeLang v0.1 + Toy Synthesizer (Week 3-8)
- [ ] EmergeLang grammar (PEG) covering property classes 1-3 (spatial, temporal, fault tolerance)
- [ ] Parser → AST → ETL compiler
- [ ] Type checker for agent schemas
- [ ] FST agent model + interpreter
- [ ] Brute-force enumerative synthesizer (no SMT, just enumerate small FSTs and simulate)
- [ ] **Demo**: synthesize symmetric consensus for 3 agents from spec
- **Paper potential**: workshop paper at SPLASH/OOPSLA workshop on emerging PL topics

### Milestone 2: Symbolic Synthesis Engine (Week 9-18)
- [ ] Z3 integration for SMT-based synthesis
- [ ] CEGIS loop for FST rules
- [ ] ETL-to-SMT encoding for safety + liveness properties
- [ ] GCP agent model
- [ ] Template library (consensus, coverage, formation templates)
- [ ] **Demo**: synthesize a consensus protocol comparable to Raft from spec
- **Paper potential**: PLDI or POPL (if Raft-comparable synthesis succeeds)

### Milestone 3: Neural-Guided Synthesis (Week 19-26)
- [ ] PyTorch Geometric GNN architecture for agent rule learning
- [ ] Differentiable ETL loss functions
- [ ] Training pipeline with curriculum learning
- [ ] Rule extraction: DiffLogic discretization + decision tree distillation
- [ ] Hybrid loop connecting neural and symbolic paths
- [ ] **Demo**: synthesize flocking rules from coverage/formation spec
- **Paper potential**: NeurIPS or ICML (neural program synthesis for emergence)

### Milestone 4: Formal Verifier (Week 27-34)
- [ ] Parameterized model checking for FST rules (cutoff computation + nuXmv)
- [ ] Lyapunov synthesis for CP rules (dReal integration)
- [ ] Statistical verification engine (parallel simulation + Bayesian bounds)
- [ ] Verification certificates
- [ ] **Demo**: formally verify synthesized consensus protocol for arbitrary N
- **Paper potential**: CAV or VMCAI (emergence verification)

### Milestone 5: Swarm Runtime (Week 35-40)
- [ ] WebAssembly simulation target with browser visualization
- [ ] Embedded code generator (C target for STM32)
- [ ] Distributed runtime (async Rust + gRPC)
- [ ] Runtime ETL monitor
- [ ] **Demo**: deploy synthesized coverage rules on 10 simulated drones in browser

### Milestone 6: Full Benchmark Evaluation (Week 41-48)
- [ ] Implement all 8 benchmark problems
- [ ] Run full evaluation pipeline
- [ ] Performance comparison with hand-designed baselines
- [ ] Scalability analysis
- [ ] **Paper**: flagship systems paper (SOSP, OSDI, or PLDI)

---

## Crate Structure

```
inverse-emergence-compiler/
├── Cargo.toml                    # Workspace root
├── README.md
├── PLAN.md
│
├── crates/
│   ├── emergelang/               # Specification language
│   │   ├── src/
│   │   │   ├── grammar.pest      # PEG grammar
│   │   │   ├── parser.rs         # PEG parser
│   │   │   ├── ast.rs            # AST types
│   │   │   ├── typecheck.rs      # Type checker
│   │   │   ├── etl.rs            # ETL (Emergence Temporal Logic) IR
│   │   │   └── compile.rs        # AST → ETL compilation
│   │   └── Cargo.toml
│   │
│   ├── agent-model/              # Agent representations
│   │   ├── src/
│   │   │   ├── fst.rs            # Finite State Transducer
│   │   │   ├── gcp.rs            # Guarded Command Program
│   │   │   ├── cp.rs             # Continuous Program
│   │   │   ├── topology.rs       # Topology definitions
│   │   │   └── interpreter.rs    # Rule interpreter for simulation
│   │   └── Cargo.toml
│   │
│   ├── synthesizer/              # Rule synthesis engine
│   │   ├── src/
│   │   │   ├── cegis.rs          # CEGIS loop
│   │   │   ├── smt_encode.rs     # ETL → SMT encoding
│   │   │   ├── templates.rs      # Rule template library
│   │   │   ├── enumerate.rs      # Brute-force enumerator (baseline)
│   │   │   └── hybrid.rs         # Neural-symbolic orchestration
│   │   └── Cargo.toml
│   │
│   ├── verifier/                 # Emergence verification
│   │   ├── src/
│   │   │   ├── parameterized.rs  # Parameterized model checking
│   │   │   ├── lyapunov.rs       # Lyapunov synthesis
│   │   │   ├── statistical.rs    # Statistical verification
│   │   │   ├── monitor.rs        # Runtime ETL monitoring
│   │   │   └── certificate.rs    # Proof certificates
│   │   └── Cargo.toml
│   │
│   ├── runtime/                  # Swarm runtime & code generation
│   │   ├── src/
│   │   │   ├── wasm.rs           # WebAssembly target
│   │   │   ├── embedded.rs       # Embedded C/Rust target
│   │   │   ├── distributed.rs    # Async Rust target
│   │   │   └── codegen.rs        # Shared code generation utilities
│   │   └── Cargo.toml
│   │
│   └── iec-cli/                  # Command-line interface
│       ├── src/
│       │   └── main.rs           # CLI entry point
│       └── Cargo.toml
│
├── neural/                       # Python: neural synthesis components
│   ├── pyproject.toml
│   ├── src/
│   │   ├── gnn_synthesizer.py    # GNN-based rule learning
│   │   ├── diff_etl_loss.py      # Differentiable ETL losses
│   │   ├── rule_extraction.py    # Neural → symbolic rule extraction
│   │   └── training.py           # Training pipeline
│   └── bridge/                   # PyO3 Rust-Python bridge
│       └── src/lib.rs
│
├── benchmarks/                   # Benchmark suite
│   ├── specs/                    # EmergeLang specifications
│   │   ├── consensus.emerge
│   │   ├── coverage.emerge
│   │   ├── flocking.emerge
│   │   ├── self_healing_ring.emerge
│   │   ├── traffic.emerge
│   │   ├── byzantine.emerge
│   │   ├── foraging.emerge
│   │   └── small_world.emerge
│   └── baselines/                # Hand-designed baseline implementations
│
├── examples/                     # Walkthrough examples
│   ├── hello_emergence/          # Simplest possible example
│   └── drone_coverage/           # Full pipeline demo
│
└── docs/
    ├── emergelang_spec.md        # Language reference
    ├── etl_semantics.md          # Formal semantics of ETL
    └── architecture.md           # System architecture details
```

---

## Key Dependencies

### Rust
- `pest` — PEG parser generator
- `z3` (via `z3-sys` + `z3`) — SMT solver
- `tokio` — async runtime for distributed target
- `wasm-bindgen` + `wgpu` — WebAssembly + GPU rendering
- `serde` — serialization
- `rayon` — parallel simulation
- `pyo3` — Python interop for neural components

### Python
- `torch` + `torch_geometric` — GNN training
- `jax` + `diffrax` — differentiable simulation (alternative to PyTorch)
- `numpy`, `scipy` — numerical computation
- `matplotlib` — visualization

### External Tools (called as subprocesses or via C FFI)
- Z3 (4.12+) — SMT solver
- nuXmv — symbolic model checker
- dReal — δ-complete SMT solver for nonlinear reals
- (optional) ByMC — parameterized model checker for threshold protocols

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| ETL model checking is undecidable for target specs | High | Critical | Tier the verification: formal for decidable fragments, statistical for the rest. Always provide statistical fallback. |
| SMT-based synthesis doesn't scale beyond tiny FSTs | High | High | Neural path as alternative; template library to constrain search; abstract-then-refine strategy |
| Neural rule extraction produces rules that fail verification | Medium | Medium | Use DiffLogic (inherently discrete); iterative refinement loop between extraction and verification |
| Cutoff results don't extend to ETL | Medium | High | Develop new cutoff theory for aggregate quantifiers; fall back to bounded verification + statistical testing |
| No single synthesized protocol matches Raft performance | Medium | Medium | Acceptable — even 80% of Raft performance with automatic synthesis is publishable. The contribution is the methodology, not beating Raft. |
