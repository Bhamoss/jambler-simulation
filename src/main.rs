mod cli;
mod tasks;
use std::{fs::create_dir_all};
use rand_chacha::ChaCha20Rng;
use rand::{RngCore, SeedableRng, thread_rng};
use clap::Clap;
use tasks::channel_occurence::channel_occurences;


fn get_rand(seed : Option<u64>) -> Box<dyn RngCore>  {
    match seed {
        Some(seed) => {
            println!("Running with seed {}.", seed);
            Box::new(ChaCha20Rng::seed_from_u64(seed))
        }
        None => {
            println!("Running without seed, pseudorandom.");
            Box::new(thread_rng())
        }
    }
}

fn main() {
    let args = cli::Args::parse();
    println!("Saving to {:?}.", args.output_dir);
    create_dir_all(&args.output_dir).expect("Could not create the given path.");

    let mut  rng = get_rand(args.seed);


    match args.task {
        cli::SimTask::All => {
            println!("Running all tasks");
            channel_occurences(&args.output_dir, &mut rng);
        }
        cli::SimTask::ChannelOccurrences => {
            println!("Running some task");
            channel_occurences(&args.output_dir, &mut rng);
        }
    }
}
