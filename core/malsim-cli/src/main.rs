//! CLI to run simulations in MAL Simulator using scenario files - the Rust
//! counterpart of `python/malsim/__main__.py`.

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use malsim_core::graph_state::TtcMode;
use malsim_core::run_simulation::run_simulation;
use malsim_core::scenario::Scenario;
use malsim_core::settings::{AttackSurfaceSettings, MalSimulatorSettings};
use malsim_core::simulator::Simulator;

const TTC_MODE_HELP: &str = "\
0: EFFORT_BASED_PER_STEP_SAMPLE\t
1: PER_STEP_SAMPLE\t
2: PRE_SAMPLE\t
3: EXPECTED_VALUE\t
4: DISABLED";

#[derive(Parser, Debug)]
#[command(name = "malsim", version)]
struct Args {
    /// Can be found in https://github.com/mal-lang/malsim-scenarios/
    scenario_file: PathBuf,

    /// If set to a path, attack graph will be dumped there
    #[arg(short, long)]
    output_attack_graph: Option<PathBuf>,

    /// If set to a seed, simulator will use it as setting
    #[arg(short, long)]
    seed: Option<u64>,

    #[arg(
        short,
        long,
        default_value_t = TtcMode::Disabled.value(),
        value_parser = clap::value_parser!(u8).range(0..=4),
        help = TTC_MODE_HELP,
    )]
    ttc_mode: u8,

    /// Maximum number of simulation steps to run
    #[arg(long)]
    max_steps: Option<u64>,
}

/// Entrypoint function of the MAL Simulator CLI
fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    let mut scenario = Scenario::load_from_file(&args.scenario_file)?;
    // As in Python, the CLI's settings replace the scenario file's
    // `sim_settings` wholesale.
    scenario.sim_settings = MalSimulatorSettings {
        seed: args.seed,
        ttc_mode: TtcMode::from_value(args.ttc_mode).ok_or("invalid TTC mode")?,
        attack_surface: AttackSurfaceSettings {
            skip_unnecessary: false,
            ..AttackSurfaceSettings::default()
        },
        ..MalSimulatorSettings::default()
    };
    let mut sim = Simulator::new(scenario.attack_graph.clone());

    if let Some(path) = &args.output_attack_graph {
        scenario
            .attack_graph
            .borrow()
            .save_to_file(Some(&scenario.model.borrow()), path)?;
    }

    run_simulation(&mut sim, &scenario, args.max_steps)?;
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
