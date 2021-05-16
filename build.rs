// Is like copy past. To avoid having to make a library
include!("src/cli.rs");
use clap::IntoApp;
use clap_generate::{
    generate_to,
    generators::{Bash, Elvish, Fish, PowerShell, Zsh},
};

fn main() {
    let mut app = Args::into_app();
    app.set_bin_name("jambler-simulation");

    let outdir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("completions/");
    generate_to::<Bash, _, _>(&mut app, "jambler-simulation", &outdir).unwrap();
    generate_to::<Fish, _, _>(&mut app, "jambler-simulation", &outdir).unwrap();
    generate_to::<Zsh, _, _>(&mut app, "jambler-simulation", &outdir).unwrap();
    generate_to::<PowerShell, _, _>(&mut app, "jambler-simulation", &outdir).unwrap();
    generate_to::<Elvish, _, _>(&mut app, "jambler-simulation", &outdir).unwrap();
}
