//! End-to-end integration tests: .emerge spec → parse → compile → synthesize → verify.

use agent_model::interpreter::Trace;
use agent_model::topology::Topology;

/// Parse a spec, compile properties, synthesize FST, verify — full pipeline.
fn run_pipeline(
    spec_src: &str,
    max_states: u32,
    test_agents: usize,
    checker: impl Fn(&Trace) -> bool + Sync + Clone,
) -> (Option<agent_model::fst::Fst>, f64) {
    // Step 1: Parse
    let program = emergelang::parse(spec_src).expect("spec should parse");

    // Step 2: Compile to ETL
    let etl_specs = emergelang::compile_to_etl(&program);
    assert!(!etl_specs.is_empty(), "should produce at least one ETL spec");

    // Step 3: Extract properties
    let emerge_decl = program
        .items
        .iter()
        .find_map(|i| match i {
            emergelang::ast::Item::Emerge(e) => Some(e),
            _ => None,
        })
        .expect("should have an emerge declaration");

    let compiled = synthesizer::properties::CompiledProperty::from_emerge_decl(emerge_decl);
    assert!(!compiled.checks.is_empty(), "should detect properties");

    // Step 4: Synthesize (enumerative, small scale)
    let checker_clone = checker.clone();
    let config = synthesizer::enumerate::EnumerateConfig {
        max_states,
        num_observations: 2,
        num_actions: 1,
        topology: Topology::Complete,
        test_agents,
        test_runs: 50,
        max_steps: 30,
        value_range: 2,
        min_success_rate: 0.80,
    };
    let result = synthesizer::enumerate::synthesize(&config, checker_clone);

    // Step 5: If found, verify statistically
    if let Some(ref fst) = result.fst {
        let verify_config = verifier::statistical::StatisticalConfig {
            num_runs: 200,
            num_agents: test_agents,
            max_steps: 50,
            value_range: 2,
            topology: Topology::Complete,
            fault_removals: 0,
        };
        let verify = verifier::statistical::verify_statistically(fst, &verify_config, checker);
        (Some(fst.clone()), verify.success_rate)
    } else {
        (None, result.best_success_rate)
    }
}

#[test]
fn test_end_to_end_consensus() {
    let spec = r#"
        agent_type Node {
            state: { value: Int }
            observe: neighbors(radius: Nat) -> Set<Node>
            act: { update }
        }
        topology Complete()
        emerge Consensus(agents: Swarm<Node>) {
            eventually globally: all_agree(agents)
        }
    "#;

    let (fst, rate) = run_pipeline(spec, 2, 3, |trace| trace.eventually_globally_agree());
    assert!(fst.is_some(), "should synthesize an FST for consensus");
    assert!(rate >= 0.5, "success rate should be reasonable, got {}", rate);
}

#[test]
fn test_end_to_end_consensus_not_degenerate() {
    let spec = r#"
        agent_type Node {
            state: { value: Int }
            observe: neighbors(radius: Nat) -> Set<Node>
            act: { update }
        }
        topology Complete()
        emerge Consensus(agents: Swarm<Node>) {
            eventually globally: all_agree(agents)
        }
    "#;

    let (fst, _) = run_pipeline(spec, 2, 3, |trace| trace.eventually_globally_agree());
    if let Some(fst) = fst {
        assert!(
            !fst.is_degenerate(),
            "synthesized FST should not be degenerate"
        );
    }
}

#[test]
fn test_parse_all_benchmarks() {
    let benchmarks = [
        include_str!("../../../benchmarks/specs/consensus.emerge"),
        include_str!("../../../benchmarks/specs/coverage.emerge"),
        include_str!("../../../benchmarks/specs/flocking.emerge"),
        include_str!("../../../benchmarks/specs/self_healing_ring.emerge"),
    ];

    for (i, src) in benchmarks.iter().enumerate() {
        let program = emergelang::parse(src).unwrap_or_else(|e| {
            panic!("benchmark {} failed to parse: {}", i, e);
        });

        // Each benchmark should have agent_type, topology, and emerge
        let has_agent = program
            .items
            .iter()
            .any(|i| matches!(i, emergelang::ast::Item::AgentType(_)));
        let has_topo = program
            .items
            .iter()
            .any(|i| matches!(i, emergelang::ast::Item::Topology(_)));
        let has_emerge = program
            .items
            .iter()
            .any(|i| matches!(i, emergelang::ast::Item::Emerge(_)));

        assert!(has_agent, "benchmark {} should have agent_type", i);
        assert!(has_topo, "benchmark {} should have topology", i);
        assert!(has_emerge, "benchmark {} should have emerge", i);

        // ETL compilation should work
        let specs = emergelang::compile_to_etl(&program);
        assert!(!specs.is_empty(), "benchmark {} should produce ETL specs", i);
    }
}

#[test]
fn test_compile_and_check_properties() {
    let spec = r#"
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

    let program = emergelang::parse(spec).unwrap();
    let emerge = program
        .items
        .iter()
        .find_map(|i| match i {
            emergelang::ast::Item::Emerge(e) => Some(e),
            _ => None,
        })
        .unwrap();

    let compiled = synthesizer::properties::CompiledProperty::from_emerge_decl(emerge);

    // Should detect: consensus, fault tolerance, convergence
    let check_types: Vec<_> = compiled
        .checks
        .iter()
        .map(|c| std::mem::discriminant(c))
        .collect();
    assert!(
        compiled.checks.iter().any(|c| matches!(
            c,
            synthesizer::properties::PropertyCheck::EventuallyGloballyAgree
        )),
        "should detect consensus property"
    );
    assert!(
        compiled.checks.iter().any(|c| matches!(
            c,
            synthesizer::properties::PropertyCheck::FaultTolerant { .. }
        )),
        "should detect fault tolerance"
    );
    assert!(
        compiled.checks.iter().any(|c| matches!(
            c,
            synthesizer::properties::PropertyCheck::ConvergeWithin { .. }
        )),
        "should detect convergence bound, but got: {:?}",
        check_types
    );
}

#[test]
fn test_cegis_pipeline() {
    let config = synthesizer::cegis::CegisConfig {
        num_states: 3,
        num_observations: 2,
        num_actions: 1,
        topology: Topology::Complete,
        test_agents: 3,
        max_steps: 20,
        value_range: 2,
        max_iterations: 10,
        tests_per_iteration: 30,
        min_success_rate: 0.70,
        initial_unroll_depth: 3,
        max_unroll_depth: 8,
    };

    let result =
        synthesizer::cegis::synthesize_cegis(&config, |trace| trace.eventually_globally_agree());

    assert!(result.fst.is_some(), "CEGIS should find an FST");
    if let Some(fst) = &result.fst {
        assert!(!fst.is_degenerate(), "CEGIS result should not be degenerate");
    }
}

#[test]
fn test_simulation_scenario() {
    let mut fst = agent_model::fst::Fst::new("test", 2, 2, 1);
    fst.add_transition(0, 0, 0, 0);
    fst.add_transition(0, 1, 1, 0);
    fst.add_transition(1, 0, 0, 0);
    fst.add_transition(1, 1, 1, 0);
    fst.set_output(0, 0);
    fst.set_output(1, 1);

    let scenario = runtime::simulation::Scenario {
        name: "test_consensus".to_string(),
        num_agents: 5,
        topology: Topology::Complete,
        initial_values: vec![0, 0, 0, 1, 1],
        max_steps: 30,
        properties: vec![verifier::monitor::MonitorProperty::AllAgree],
    };

    let result = runtime::simulation::run_scenario(&fst, &scenario);
    assert_eq!(result.name, "test_consensus");
    assert!(result.trace_length > 0);
}
