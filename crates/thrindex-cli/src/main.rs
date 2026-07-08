//! `thrindex-cli` — the standalone Rust binary.
//!
//! Peer consumer of `thrindex-sim`.  The Python `thrindex._cli` entrypoint is a
//! separate peer consumer that calls the same library via `PyO3` — neither calls the
//! other (correction 3 / ARCHITECTURE.md layer law).
//!
//! Usage:
//!   thrindex-cli run <model.thx> [--seed N] [--threads N]
//!   thrindex-cli doctor [--check <model.thx>] [--verbose]
//!   thrindex-cli targets

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
