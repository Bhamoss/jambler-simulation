use std::path::PathBuf;

use clap::Clap;

#[derive(Clap)]
pub struct Args {
    /// Which tasks to run
    #[clap(long, short, arg_enum, default_value = "all")]
    pub task: SimTask,
    /// Output directory
    #[clap(long, short, takes_value = true, default_value = "plots")]
    pub output_dir: PathBuf,
    /// The positive seed for the simulations. Random seed if not given.
    #[clap(long, short, takes_value = true)]
    pub seed: Option<u64>,
    /// The physical chance of capturing a packet when a sniffer is listening for it on a channel.
    #[clap(long, short, takes_value = true, default_value = "0.9")]
    pub capture_chance: f64,
    /// The number of sniffers.
    #[clap(long, short, takes_value = true, default_value = "5")]
    pub nb_sniffers: u8,
}

#[derive(Clap, Copy, Clone)]
pub enum SimTask {
    /// Run all tasks
    All,
    /// Channel occurrences
    ChannelOccurrences,
    /// False positive recovery trick
    ChannelRecovery,
    ConnectionInterval,
}

//fn validate_output_dir(path:& str) -> Result<(), String> {
//    std::fs::create_dir_all(path).map_err(|x| x.to_string())?;
//    Ok(())
//}

//fn validate_seed(seed:& str) -> Result<(), String> {
//    seed.parse::<u64>().map_err(|x| x.to_string())?;
//    Ok(())
//}
