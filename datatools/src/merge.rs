use anyhow::Context;
use indicatif::{ProgressBar, ProgressStyle};
use std::{path::PathBuf, time::Duration};
use tokio::{
    fs::{File, OpenOptions},
    io,
};

use crate::shuffle::shuffle;

#[derive(clap::Args)]
pub struct Args {
    #[clap(help("Input data files"))]
    inputs: Vec<PathBuf>,
    #[clap(short('o'))]
    output: PathBuf,
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    let mut output_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(true)
        .open(&args.output)
        .await
        .with_context(|| format!("failed to open output path `{}`", args.output.display()))?;

    let progress = ProgressBar::new(args.inputs.len() as u64)
        .with_style(
            ProgressStyle::with_template(
                "{spinner} [{elapsed_precise:.yellow}] [{bar:20}] {msg} {pos}/{len} files merged",
            )
            .unwrap()
            .progress_chars("##-"),
        )
        .with_message("merging files...");
    progress.enable_steady_tick(Duration::from_millis(50));
    for input_path in &args.inputs {
        let mut input_file = File::open(input_path)
            .await
            .with_context(|| format!("failed to open input file `{}`", input_path.display()))?;
        io::copy(&mut input_file, &mut output_file).await?;
        progress.inc(1);
    }
    progress.finish();

    shuffle(output_file, None).await
}
