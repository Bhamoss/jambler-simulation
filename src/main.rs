mod cli;
mod tasks;
use clap::Clap;
use itertools::Itertools;
use rand::{thread_rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::{fs::create_dir_all, path::PathBuf, sync::{Arc, Mutex}, thread, time::Duration};
use tasks::channel_occurrence::channel_occurrences;
use indicatif::MultiProgress;

use crate::tasks::{channel_recovery::channel_recovery, conn_interval::conn_interval};

fn get_rand(seed: Option<u64>) -> Box<dyn RngCore> {
    match seed {
        Some(seed) => {
            //println!("Running with seed {}.", seed);
            Box::new(ChaCha20Rng::seed_from_u64(seed))
        }
        None => {
            //println!("Running without seed, pseudorandom.");
            Box::new(thread_rng())
        }
    }
}

#[derive(Debug)]
pub struct SimulationParameters<R: RngCore + Send + Sync> {
    pub nb_sniffers: u8,
    pub capture_chance: f64,
    pub rng: R,
    pub output_dir: PathBuf,
}

impl SimulationParameters<ChaCha20Rng> {
    pub fn new(nb_sniffers: u8, capture_chance: f64, output_dir: PathBuf, random_u64: u64) -> Self {
        SimulationParameters {
            nb_sniffers,
            capture_chance,
            rng: ChaCha20Rng::seed_from_u64(random_u64),
            output_dir,
        }
    }
}

impl<R: RngCore + Send + Sync> SimulationParameters<R> {
    fn gen_new(other: &mut SimulationParameters<R>) -> SimulationParameters<ChaCha20Rng> {
        SimulationParameters {
            nb_sniffers: other.nb_sniffers,
            capture_chance: other.capture_chance,
            rng: ChaCha20Rng::seed_from_u64(other.rng.next_u64()),
            output_dir: other.output_dir.clone(),
        }
    }
}

trait Task: Fn(SimulationParameters<ChaCha20Rng>, Arc<Mutex<MultiProgress>>) + Send + Sync {
    // ...
}

impl<T: Fn(SimulationParameters<ChaCha20Rng>, Arc<Mutex<MultiProgress>>) + Send + Sync> Task for T {}

pub struct TaskInstance {
    params: SimulationParameters<ChaCha20Rng>,
    f: Box<dyn Task>,
    bars : Arc<Mutex<MultiProgress>>
}

fn run_tasks<R: RngCore + Send + Sync>(
    tasks: Vec<Box<dyn Task>>,
    mut params: SimulationParameters<R>,
    bars: Arc<Mutex<MultiProgress>>
) {
    // Generate their parameters
    let task_instances = tasks
        .into_iter()
        .map(|f| TaskInstance {
            params: SimulationParameters::gen_new(&mut params),
            f,
            bars: Arc::clone(&bars)
        })
        .collect_vec();
    // Run all tasks in parallel
    task_instances
        .into_par_iter()
        .for_each(|ti| (ti.f)(ti.params, ti.bars));
}

fn main() {
    let args = cli::Args::parse();
    //println!("Saving to {:?}.", args.output_dir);
    create_dir_all(&args.output_dir).expect("Could not create the given path.");

    let mut rng = get_rand(args.seed);
    let params = SimulationParameters::new(
        args.nb_sniffers,
        args.capture_chance,
        args.output_dir,
        rng.next_u64(),
    );

    let task = args.task;
    let progress = Arc::new(Mutex::new(MultiProgress::new()));
    let p = Arc::clone(&progress);
    rayon::join( move || 
    {match task {
        cli::SimTask::All => {
            //println!("Running all tasks");
            // List tasks
            let tasks: Vec<Box<dyn Task>> =
                vec![Box::new(&channel_occurrences), Box::new(&channel_recovery), Box::new(&conn_interval)];
            run_tasks(tasks, params, progress);
        }
        cli::SimTask::ChannelOccurrences => {
            //println!("Running channel occurrences");
            channel_occurrences(params, progress);
        }
        cli::SimTask::ChannelRecovery => {
            //println!("Running channel recovery");
            channel_recovery(params, progress);
        }
        cli::SimTask::ConnectionInterval => {
            conn_interval(params, progress);
        }
    }},
    || {
        thread::sleep(Duration::from_millis(1000));
        p.lock().unwrap().join_and_clear().unwrap();
    });
}
