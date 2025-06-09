use std::{io::SeekFrom, mem, path::PathBuf};
use anyhow::Context;
use dama::{Color, Outcome};
use dataformat::PackedSample;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro128PlusPlus;
use tokio::{fs::File, io::{AsyncReadExt, AsyncSeekExt}};

#[derive(clap::Args)]
pub struct Args {
    #[clap(help("File to show random samples from."))]
    file: PathBuf,
    #[clap(short('s'), long("samples"), default_value_t = 16)]
    samples: u32,
    #[clap(short('S'), long("seed"))]
    seed: Option<u64>,
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    let mut file = File::open(&args.file)
        .await
        .with_context(|| format!("failed to open file `{}`", args.file.display()))?;

    let step = mem::size_of::<PackedSample>() as u64;
    let positions = file.seek(SeekFrom::End(0)).await? / step;

    let mut rng = if let Some(seed) = args.seed {
        Xoshiro128PlusPlus::seed_from_u64(seed)
    } else {
        Xoshiro128PlusPlus::from_os_rng()
    }; 

    for n in 0..args.samples {
        let position = rng.random_range(0..positions);
        file.seek(SeekFrom::Start(position * step)).await?;

        let mut sample = PackedSample::default();
        file.read_exact(bytemuck::bytes_of_mut(&mut sample)).await?;

        let sample = sample.unpack()?;
        println!("{}\n", sample.position);
        println!("FEN: {}", sample.position.fen());
        println!("Side to move: {}", sample.position.side_to_move());
        println!("Outcome: {} ({})", sample.outcome, match sample.outcome {
            Outcome::Winner(Color::White) => "white wins",
            Outcome::Winner(Color::Black) => "black wins",
            Outcome::Draw => "draw",
        });
        if let Some(eval) = sample.eval {
            println!("Evaluation: {}", eval);
        }

        if n != args.samples - 1 {
            println!("\n———————————————————\n");
        }
    }

    Ok(())
}

