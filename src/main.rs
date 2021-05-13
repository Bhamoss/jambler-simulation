mod cli;
use clap::Clap;


fn main() {
    let args = cli::Args::parse();
    match args.task {
        cli::SimTask::All => {
            println!("Running all tasks");
        }
        cli::SimTask::SomeTask => {
            println!("Running some task");
        }
    }
}
