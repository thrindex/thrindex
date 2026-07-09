//! `thrindex-cli` — the standalone Rust binary.
//!
//! Peer consumer of `thrindex-sim` and `conformance`.  The Python `thrindex._cli`
//! entrypoint is a separate peer consumer that calls the same libraries via `PyO3` —
//! neither calls the other (correction 3 / ARCHITECTURE.md layer law).
//!
//! Usage:
//!   thrindex run <model.thx> [--seed N] [--threads N]
//!   thrindex bench --conformance --target sim [--artifact F] [--fixtures D]
//!   thrindex doctor [--check <model.thx>] [--verbose]
//!   thrindex targets

#![deny(unsafe_code)]
#![warn(clippy::pedantic)]

use clap::{Parser, Subcommand};

mod cmd;

#[derive(Parser, Debug)]
#[command(name = "thrindex", about = "The neuromorphic infrastructure SDK")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Load a `.thx` artifact and run the behavioral simulator.
    Run(cmd::run::RunArgs),
    /// Run the conformance suite or performance benchmarks.
    Bench(cmd::bench::BenchArgs),
    /// Diagnose the environment.
    Doctor(cmd::doctor::DoctorArgs),
    /// List available simulation targets.
    Targets(cmd::targets::TargetsArgs),
}

fn main() {
    let cli = Cli::parse();

    let result = match &cli.command {
        Commands::Run(args) => match cmd::run::run(args) {
            Ok(transcript) => {
                print!("{transcript}");
                Ok(())
            }
            Err(e) => Err(e.to_string()),
        },
        Commands::Bench(args) => match cmd::bench::run(args) {
            Ok(report) => {
                print!("{report}");
                Ok(())
            }
            Err(e) => Err(e),
        },
        Commands::Doctor(args) => {
            print!("{}", cmd::doctor::run(args));
            Ok(())
        }
        Commands::Targets(args) => {
            print!("{}", cmd::targets::run(args));
            Ok(())
        }
    };

    if let Err(msg) = result {
        eprintln!("{msg}");
        std::process::exit(1);
    }
}
