use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use miette::{miette, LabeledSpan, Report};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "iec",
    about = "Inverse Emergence Compiler — synthesize local rules from global emergence specs"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse an EmergeLang specification file and show its structure
    Parse {
        /// Path to the .emerge file
        file: String,
    },

    /// Synthesize local rules from a specification
    Synthesize {
        /// Path to the .emerge file
        file: String,

        /// Maximum number of FST states to try
        #[arg(long, default_value = "3")]
        max_states: u32,

        /// Number of agents for testing
        #[arg(long, default_value = "5")]
        agents: usize,

        /// Minimum success rate to accept (0.0 - 1.0)
        #[arg(long, default_value = "0.95")]
        min_rate: f64,

        /// Synthesis engine to use
        #[arg(long, default_value = "cegis")]
        engine: SynthesisEngine,
    },

    /// Run the full demo: parse, synthesize, simulate, verify
    Demo,
}

#[derive(Debug, Clone, ValueEnum)]
enum SynthesisEngine {
    /// Brute-force enumeration (slow but exhaustive)
    Enumerate,
    /// CEGIS with Z3 SMT solver (faster, smarter)
    Cegis,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Parse { file } => cmd_parse(&file),
        Commands::Synthesize {
            file,
            max_states,
            agents,
            min_rate,
            engine,
        } => cmd_synthesize(&file, max_states, agents, min_rate, engine),
        Commands::Demo => cmd_demo(),
    }
}

fn cmd_parse(file: &str) -> Result<()> {
    let source = std::fs::read_to_string(file)?;
    let program = match emergelang::parse(&source) {
        Ok(p) => p,
        Err(e) => {
            let report = make_parse_diagnostic(file, &source, &e);
            eprintln!("{:?}", report);
            return Err(anyhow::anyhow!("Parse failed"));
        }
    };

    println!("Parsed {} items:", program.items.len());
    for item in &program.items {
        match item {
            emergelang::ast::Item::AgentType(a) => {
                println!("  agent_type {} ({} state fields, {} actions)",
                    a.name, a.state_fields.len(), a.actions.len());
            }
            emergelang::ast::Item::Topology(t) => {
                println!("  topology {} ({} params)", t.name, t.params.len());
            }
            emergelang::ast::Item::Emerge(e) => {
                println!("  emerge {} ({} clauses)", e.name, e.clauses.len());

                // Compile properties and show what was detected
                let compiled = synthesizer::properties::CompiledProperty::from_emerge_decl(e);
                println!("    Detected properties:");
                for check in &compiled.checks {
                    println!("      - {:?}", check);
                }
            }
        }
    }

    // Compile to ETL
    let etl_specs = emergelang::compile_to_etl(&program);
    for spec in &etl_specs {
        println!("\nETL Spec: {}", spec.name);
        println!("  Safety properties: {}", spec.safety.len());
        println!("  Liveness properties: {}", spec.liveness.len());
        println!("  Fault tolerance: {}", spec.fault_tolerance.len());
        if let Some(bound) = spec.convergence_bound {
            println!("  Convergence bound: {} steps", bound);
        }
    }

    Ok(())
}

fn cmd_synthesize(file: &str, max_states: u32, agents: usize, min_rate: f64, engine: SynthesisEngine) -> Result<()> {
    let source = std::fs::read_to_string(file)?;
    let program = match emergelang::parse(&source) {
        Ok(p) => p,
        Err(e) => {
            let report = make_parse_diagnostic(file, &source, &e);
            eprintln!("{:?}", report);
            return Err(anyhow::anyhow!("Parse failed"));
        }
    };

    // Find the emerge declaration and compile properties
    let emerge_decl = program.items.iter().find_map(|i| match i {
        emergelang::ast::Item::Emerge(e) => Some(e),
        _ => None,
    }).ok_or_else(|| anyhow::anyhow!("No emerge declaration found in spec"))?;

    let compiled_prop = synthesizer::properties::CompiledProperty::from_emerge_decl(emerge_decl);

    println!("Synthesizing local rules for: {}", compiled_prop.name);
    println!("  Engine: {:?}", engine);
    println!("  Max FST states: {}", max_states);
    println!("  Test agents: {}", agents);
    println!("  Min success rate: {:.0}%", min_rate * 100.0);
    println!("  Properties:");
    for check in &compiled_prop.checks {
        println!("    - {:?}", check);
    }
    println!();

    // Create the property checker closure
    let prop_for_check = compiled_prop.clone();
    let checker = move |trace: &agent_model::interpreter::Trace| -> bool {
        prop_for_check.check(trace)
    };

    let (fst_result, elapsed, extra_info) = match engine {
        SynthesisEngine::Enumerate => {
            let config = synthesizer::enumerate::EnumerateConfig {
                max_states,
                num_observations: 2,
                num_actions: 1,
                topology: agent_model::topology::Topology::Complete,
                test_agents: agents,
                test_runs: 100,
                max_steps: 50,
                value_range: 2,
                min_success_rate: min_rate,
            };
            let result = synthesizer::enumerate::synthesize(&config, checker);
            let info = format!("Candidates evaluated: {}", result.candidates_evaluated);
            (result.fst, result.elapsed_seconds, info)
        }
        SynthesisEngine::Cegis => {
            let config = synthesizer::cegis::CegisConfig {
                num_states: max_states,
                num_observations: 2,
                num_actions: 1,
                topology: agent_model::topology::Topology::Complete,
                test_agents: agents,
                max_steps: 50,
                value_range: 2,
                max_iterations: 50,
                tests_per_iteration: 100,
                min_success_rate: min_rate,
                initial_unroll_depth: 3,
                max_unroll_depth: 10,
            };
            let result = synthesizer::cegis::synthesize_cegis(&config, checker);
            let info = format!(
                "CEGIS iterations: {}, counterexamples: {}",
                result.iterations, result.counterexamples_used
            );
            (result.fst, result.elapsed_seconds, info)
        }
    };

    println!("Synthesis complete:");
    println!("  {}", extra_info);
    println!("  Time: {:.2}s", elapsed);

    if let Some(fst) = &fst_result {
        println!("\nSynthesized FST:");
        println!("  States: {}", fst.num_states);
        println!("  Degenerate: {}", fst.is_degenerate());
        println!("  Transitions:");
        let mut transitions: Vec<_> = fst.transitions.iter().collect();
        transitions.sort_by_key(|((s, o), _)| (*s, *o));
        for ((from, obs), (to, act)) in &transitions {
            println!("    ({}, obs={}) -> ({}, act={})", from, obs, to, act);
        }
        println!("  Outputs:");
        let mut outputs: Vec<_> = fst.output.iter().collect();
        outputs.sort_by_key(|(s, _)| *s);
        for (state, value) in &outputs {
            println!("    state {} -> value {}", state, value);
        }

        // Verify with statistical testing
        println!("\nStatistical Verification:");
        let verify_config = verifier::statistical::StatisticalConfig {
            num_runs: 1000,
            num_agents: agents,
            max_steps: 100,
            value_range: 2,
            topology: agent_model::topology::Topology::Complete,
            fault_removals: 0,
        };

        let prop_for_verify = compiled_prop.clone();
        let verify_result = verifier::statistical::verify_statistically(
            fst,
            &verify_config,
            move |trace| prop_for_verify.check(trace),
        );

        println!("  Success rate: {:.1}% ({}/{})",
            verify_result.success_rate * 100.0,
            verify_result.successful_runs,
            verify_result.total_runs);
        println!("  99% confidence lower bound: {:.1}%",
            verify_result.confidence_lower_bound * 100.0);
        if let Some(avg) = verify_result.avg_convergence_step {
            println!("  Avg convergence step: {:.1}", avg);
        }
        if let Some(max) = verify_result.max_convergence_step {
            println!("  Max convergence step: {}", max);
        }
        if !verify_result.failure_examples.is_empty() {
            println!("  Sample failures:");
            for (i, ex) in verify_result.failure_examples.iter().enumerate().take(3) {
                println!("    #{}: {:?}", i + 1, ex);
            }
        }

        // Save synthesized FST
        let json_transitions: std::collections::HashMap<String, (u32, u32)> = fst
            .transitions
            .iter()
            .map(|((s, o), (ns, a))| (format!("{},{}", s, o), (*ns, *a)))
            .collect();
        let json_output: std::collections::HashMap<String, i64> = fst
            .output
            .iter()
            .map(|(s, v)| (s.to_string(), *v))
            .collect();
        let json_fst = serde_json::json!({
            "name": fst.name,
            "num_states": fst.num_states,
            "num_observations": fst.num_observations,
            "num_actions": fst.num_actions,
            "initial_state": fst.initial_state,
            "transitions": json_transitions,
            "output": json_output,
            "degenerate": fst.is_degenerate(),
        });
        let output = serde_json::to_string_pretty(&json_fst)?;
        let output_path = file.replace(".emerge", ".fst.json");
        std::fs::write(&output_path, &output)?;
        println!("\nSaved FST to: {}", output_path);
    } else {
        println!("\nNo satisfying FST found.");
    }

    Ok(())
}

fn cmd_demo() -> Result<()> {
    println!("=== Inverse Emergence Compiler Demo ===\n");

    // Step 1: Parse a spec
    println!("Step 1: Parsing EmergeLang specification\n");
    let spec_src = r#"
        agent_type Node {
            state: { value: Int, decided: Bool }
            observe: neighbors_majority(radius: Nat) -> Int
            act: { update_value }
        }

        topology Complete()

        emerge Consensus(agents: Swarm<Node>) {
            eventually globally: all_agree(agents)
            forall d in agents:
                without(d): eventually globally: all_agree(agents)
            converge_within: 100 steps
        }
    "#;

    let program = emergelang::parse(spec_src)
        .map_err(|e| anyhow::anyhow!("Parse error: {}", e))?;
    let emerge = program.items.iter().find_map(|i| match i {
        emergelang::ast::Item::Emerge(e) => Some(e),
        _ => None,
    }).unwrap();

    let compiled_prop = synthesizer::properties::CompiledProperty::from_emerge_decl(emerge);
    println!("  Spec: {}", compiled_prop.name);
    println!("  Properties:");
    for check in &compiled_prop.checks {
        println!("    - {:?}", check);
    }

    // Step 2: Synthesize with CEGIS
    println!("\nStep 2: CEGIS Synthesis (Z3 SMT solver)\n");
    let prop_for_synth = compiled_prop.clone();
    let cegis_config = synthesizer::cegis::CegisConfig {
        num_states: 3,
        num_observations: 2,
        num_actions: 1,
        topology: agent_model::topology::Topology::Complete,
        test_agents: 5,
        max_steps: 30,
        value_range: 2,
        max_iterations: 30,
        tests_per_iteration: 50,
        min_success_rate: 0.85,
        initial_unroll_depth: 3,
        max_unroll_depth: 10,
    };

    let result = synthesizer::cegis::synthesize_cegis(&cegis_config, move |trace| {
        prop_for_synth.check(trace)
    });

    println!("  CEGIS iterations: {}", result.iterations);
    println!("  Counterexamples used: {}", result.counterexamples_used);
    println!("  Time: {:.2}s", result.elapsed_seconds);

    let fst = match result.fst {
        Some(f) => f,
        None => {
            println!("  No satisfying FST found!");
            return Ok(());
        }
    };

    println!("  Success rate: {:.1}%", result.final_success_rate * 100.0);
    println!("  Degenerate: {}", fst.is_degenerate());

    // Step 3: Show the FST
    println!("\nStep 3: Synthesized Local Rule\n");
    println!("  States: {}", fst.num_states);
    let mut transitions: Vec<_> = fst.transitions.iter().collect();
    transitions.sort_by_key(|((s, o), _)| (*s, *o));
    for ((from, obs), (to, _act)) in &transitions {
        println!("    state={}, obs={} -> state={}", from, obs, to);
    }
    println!("  Outputs:");
    let mut outputs: Vec<_> = fst.output.iter().collect();
    outputs.sort_by_key(|(s, _)| *s);
    for (state, value) in &outputs {
        println!("    state {} -> value {}", state, value);
    }

    // Step 4: Run a sample simulation
    println!("\nStep 4: Sample Simulation (5 agents, initial=[0,1,0,1,0])\n");
    let trace = agent_model::interpreter::simulate(
        &fst,
        5,
        &agent_model::topology::Topology::Complete,
        &[0, 1, 0, 1, 0],
        20,
    );
    runtime::simulation::print_trace(&trace);

    // Check with compiled properties
    let prop_for_detail = compiled_prop.clone();
    let details = prop_for_detail.check_detailed(&trace);
    println!("\n  Property check results:");
    for (check, passed, detail) in &details {
        println!("    [{:}] {:?}: {}", if *passed { "PASS" } else { "FAIL" }, check, detail);
    }

    // Step 5: Statistical verification
    println!("\nStep 5: Statistical Verification (1000 runs, 10 agents)\n");
    let verify_config = verifier::statistical::StatisticalConfig {
        num_runs: 1000,
        num_agents: 10,
        max_steps: 100,
        value_range: 2,
        topology: agent_model::topology::Topology::Complete,
        fault_removals: 0,
    };

    let prop_for_verify = compiled_prop.clone();
    let verify = verifier::statistical::verify_statistically(
        &fst,
        &verify_config,
        move |trace| prop_for_verify.check(trace),
    );

    println!("  Total runs: {}", verify.total_runs);
    println!("  Success rate: {:.1}% ({}/{})",
        verify.success_rate * 100.0, verify.successful_runs, verify.total_runs);
    println!("  99% confidence: P(spec satisfied) >= {:.1}%",
        verify.confidence_lower_bound * 100.0);
    if let Some(avg) = verify.avg_convergence_step {
        println!("  Avg convergence: {:.1} steps", avg);
    }

    println!("\n=== Demo Complete ===");
    Ok(())
}

/// Create a pretty miette diagnostic from a parse error.
fn make_parse_diagnostic(
    filename: &str,
    source: &str,
    error: &emergelang::parser::ParseError,
) -> Report {
    match error {
        emergelang::parser::ParseError::Grammar(msg) => {
            // Try to extract line/column from pest error message
            if let Some(offset) = find_error_offset(msg, source) {
                miette!(
                    labels = vec![LabeledSpan::at(offset..offset + 1, "parse error here")],
                    help = "check EmergeLang syntax",
                    "Failed to parse {}",
                    filename,
                )
                .with_source_code(source.to_string())
            } else {
                miette!("Failed to parse {}: {}", filename, msg)
            }
        }
        emergelang::parser::ParseError::Unexpected(msg) => {
            miette!(
                help = "this may indicate a malformed specification",
                "Unexpected structure in {}: {}",
                filename,
                msg
            )
        }
    }
}

/// Try to extract byte offset from a pest error message containing line/col info.
fn find_error_offset(msg: &str, source: &str) -> Option<usize> {
    // Pest errors look like: " --> 3:5" or "at line 3, column 5"
    let re_arrow = regex_lite::Regex::new(r"-->\s*(\d+):(\d+)").ok()?;
    if let Some(caps) = re_arrow.captures(msg) {
        let line: usize = caps.get(1)?.as_str().parse().ok()?;
        let col: usize = caps.get(2)?.as_str().parse().ok()?;
        return line_col_to_offset(source, line, col);
    }
    None
}

fn line_col_to_offset(source: &str, line: usize, col: usize) -> Option<usize> {
    let mut current_line = 1;
    let mut offset = 0;
    for ch in source.chars() {
        if current_line == line {
            if col <= 1 {
                return Some(offset);
            }
            // Walk col characters
            let mut c = 1;
            for ch2 in source[offset..].chars() {
                if c >= col {
                    return Some(offset);
                }
                offset += ch2.len_utf8();
                c += 1;
            }
            return Some(offset);
        }
        if ch == '\n' {
            current_line += 1;
        }
        offset += ch.len_utf8();
    }
    None
}
