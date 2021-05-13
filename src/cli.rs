use clap::Clap;

#[derive(Clap)]
pub struct Args {
    /// Print output in a format
    #[clap(long, short, arg_enum, default_value = "all")]
    pub task: SimTask,
}

#[derive(Clap, Copy, Clone)]
pub enum SimTask {
    All,
    SomeTask,
}