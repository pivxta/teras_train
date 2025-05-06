use anyhow::Context;
use core::mem;
use dataformat::PackedSample;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, seq::SliceRandom};
use std::{
    io::SeekFrom,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{
    fs::{File, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader, BufWriter},
    sync::mpsc::{UnboundedSender, unbounded_channel},
};

#[derive(clap::Args)]
pub struct Args {
    #[clap(help("Input data file."))]
    input: PathBuf,
    #[clap(short('o'))]
    output: Option<PathBuf>,
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    let input_file = OpenOptions::new()
        .create(false)
        .read(true)
        .write(args.output.is_none())
        .open(&args.input)
        .await
        .with_context(|| format!("failed to open file `{}`", args.input.display()))?;

    shuffle(input_file, args.output.as_deref()).await
}

const SUBFILE_SIZE: u64 = 2097152;

pub async fn shuffle(mut input_file: File, output_path: Option<&Path>) -> anyhow::Result<()> {
    input_file.seek(SeekFrom::Start(0)).await?;

    let progress = ProgressBar::no_length()
        .with_style(
            ProgressStyle::with_template(
                "{spinner} [{elapsed_precise:.yellow}] [{bar:20}] {msg} {pos}/{len} blocks done. ",
            )
            .unwrap()
            .progress_chars("##-"),
        )
        .with_message("shuffling positions...");
    progress.enable_steady_tick(Duration::from_millis(50));

    let (subfiles, remaining, positions) = divide_and_shuffle(&progress, &mut input_file).await?;

    let output_file = if let Some(output_path) = output_path {
        File::create(output_path)
            .await
            .with_context(|| format!("failed to open file `{}`", output_path.display()))?
    } else {
        input_file.seek(SeekFrom::Start(0)).await?;
        input_file
    };

    let progress = ProgressBar::new(positions)
        .with_style(ProgressStyle::with_template("{spinner} [{elapsed_precise:.yellow}] [{bar:20}] {msg} {pos}/{len} positions written.")
            .unwrap()
            .progress_chars("##-"))
        .with_message("writing data to output file...");
    progress.enable_steady_tick(Duration::from_millis(50));

    let (send, mut recv) = unbounded_channel();
    let task = tokio::spawn(sample_subfiles(subfiles, remaining, positions, send));

    let mut writer = BufWriter::new(output_file);
    while let Some(sample) = recv.recv().await {
        writer.write_all(bytemuck::bytes_of(&sample)).await?;
        progress.inc(1);
    }
    writer.flush().await?;
    progress.finish();
    task.await?
}

async fn divide_and_shuffle(
    progress: &ProgressBar,
    file: &mut File,
) -> anyhow::Result<(Vec<File>, Vec<u64>, u64)> {
    let positions = file.seek(SeekFrom::End(0)).await? / mem::size_of::<PackedSample>() as u64;
    file.rewind().await?;

    let mut positions_remaining = positions;
    let subfiles = positions.div_ceil(SUBFILE_SIZE);
    progress.set_length(subfiles);

    let mut tempfiles: Vec<File> = (0..subfiles)
        .map(|_| Ok(File::from_std(tempfile::tempfile()?)))
        .collect::<Result<Vec<_>, anyhow::Error>>()?;
    let mut remaining = Vec::new();

    for tempfile in tempfiles.iter_mut() {
        let subfile_positions = positions_remaining.min(SUBFILE_SIZE);
        let mut subfile = vec![PackedSample::default(); subfile_positions as usize];
        file.read_exact(bytemuck::cast_slice_mut(&mut subfile))
            .await?;
        subfile.shuffle(&mut rand::rng());
        tempfile.write_all(bytemuck::cast_slice(&subfile)).await?;
        remaining.push(subfile_positions);
        positions_remaining -= subfile_positions;
        progress.inc(1);
    }
    for tempfile in tempfiles.iter_mut() {
        tempfile.flush().await?;
        tempfile.sync_all().await?;
        tempfile.seek(SeekFrom::Start(0)).await?;
    }
    progress.finish();

    Ok((tempfiles, remaining, positions))
}

async fn sample_subfiles(
    tempfiles: Vec<File>,
    mut remaining: Vec<u64>,
    positions: u64,
    send: UnboundedSender<PackedSample>,
) -> anyhow::Result<()> {
    let mut tempfiles: Vec<_> = tempfiles.into_iter().map(BufReader::new).collect();
    let subfiles = tempfiles.len();
    let mut remaining_subfiles = subfiles;
    let last_subfile_size = positions % SUBFILE_SIZE;
    let last_subfile_idx = positions - last_subfile_size;

    while remaining_subfiles > 0 {
        let rand = rand::rng().random_range(0..positions);
        let idx = if rand < last_subfile_idx {
            (rand / SUBFILE_SIZE) as usize
        } else {
            subfiles - 1
        };

        if remaining[idx] == 0 {
            continue;
        }

        let mut sample = PackedSample::default();
        tempfiles[idx]
            .read_exact(bytemuck::bytes_of_mut(&mut sample))
            .await?;
        remaining[idx] -= 1;

        if remaining[idx] == 0 {
            remaining_subfiles -= 1;
        }

        send.send(sample)?;
    }

    Ok(())
}
