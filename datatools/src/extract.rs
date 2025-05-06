use anyhow::Context;
use core::str;
use dama::{Outcome, Position, SanMove, pgn};
use dataformat::{PackedSample, Sample};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    mem,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::Duration,
};

use crate::shuffle::shuffle;

#[derive(clap::Args)]
pub struct Args {
    #[clap(help("Input PGN files."))]
    inputs: Vec<PathBuf>,
    #[clap(short('o'), default_value("output.bin"))]
    output: PathBuf,
    #[clap(short('a'), long("append"))]
    append: bool,
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    let output_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(!args.append)
        .append(args.append)
        .open(&args.output)
        .with_context(|| format!("failed to open output path `{}`", args.output.display()))?;

    let (send, recv) = mpsc::channel();
    let reader_progress = MultiProgress::new();
    let _reader_threads = args
        .inputs
        .iter()
        .map(|path| -> Result<_, anyhow::Error> {
            let file = File::open(path)
                .with_context(|| format!("failed to open input file `{}`", path.display()))?;
            let path = path.clone();
            let send = send.clone();
            let progress = reader_progress.clone();
            Ok(thread::spawn(move || {
                read_games(&path, file, send, progress)
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(send);

    let mut writer = BufWriter::new(&output_file);
    let mut positions_written = 0;
    while let Ok(sample) = recv.recv() {
        positions_written += 1;
        writer.write_all(bytemuck::bytes_of(&sample))?;
    }
    writer.flush()?;
    drop(writer);

    println!("{} positions written", positions_written);

    shuffle(output_file.into(), None).await
}

fn read_games(
    path: &Path,
    file: File,
    send: mpsc::Sender<PackedSample>,
    multi_progress: MultiProgress,
) {
    let progress = ProgressBar::new_spinner()
        .with_message(format!("reading games from `{}...`", path.display()))
        .with_style(
            ProgressStyle::with_template(
                "{spinner} [{elapsed_precise:.yellow}] {msg} {human_pos} games read",
            )
            .unwrap(),
        );
    progress.enable_steady_tick(Duration::from_millis(100));
    multi_progress.add(progress.clone());

    let mut visitor = GameVisitor::default();
    let mut reader = pgn::Reader::new(BufReader::new(file));
    loop {
        match reader.visit_game(&mut visitor) {
            Ok(true) => {
                for sample in visitor.take_buffer() {
                    send.send(sample).expect("failed to send sample");
                }
            }
            Ok(false) => {
                progress.finish();
                break;
            }
            Err(err) if !err.is_recoverable() => {
                progress.finish();
                eprintln!("unrecoverable PGN error: {}", err);
                break;
            }
            Err(pgn::Error::Parse(err)) => {
                progress.println(format!("parsing error while reading PGN: {}", err));
            }
            Err(pgn::Error::Visitor(err)) => {
                progress.println(format!("error while reading PGN: {:#}", err));
            }
        }
        progress.inc(1);
    }
}

#[derive(Default)]
struct GameVisitor {
    buffer: Vec<PackedSample>,
    skip: bool,
    position: Position,
    outcome: Option<Outcome>,
    eval: Option<i16>,

    positions_written: u32,
    positions_seen: u32,
    games_read: u32,
    games_skipped: u32,
}

impl pgn::Visitor for GameVisitor {
    type Error = anyhow::Error;

    fn prepare(&mut self) {
        self.position = Position::new_initial();
        self.eval = None;
    }

    fn visit_tag_pair(&mut self, name: &str, value: &str) -> anyhow::Result<()> {
        match name {
            "FEN" => self.position = Position::from_fen(value)?,
            "Result" if value == "*" => self.outcome = None,
            "Result" => self.outcome = Some(value.parse()?),
            "Termination" if value != "normal" => self.skip = true,
            _ => {}
        }
        Ok(())
    }

    fn enter_game(&mut self) -> pgn::ControlFlow {
        if self.skip || self.outcome.is_none() {
            self.games_skipped += 1;
            pgn::ControlFlow::Skip
        } else {
            self.games_read += 1;
            pgn::ControlFlow::Continue
        }
    }

    fn enter_variation(&mut self) -> pgn::ControlFlow {
        pgn::ControlFlow::Skip
    }

    fn visit_move(&mut self, _number: Option<u32>, mv: SanMove) -> anyhow::Result<()> {
        if !self.position.is_in_check() && !mv.is_capture() {
            if let Some(eval) = self.eval {
                self.write(eval)?;
            }
        }

        self.position
            .play(&mv)
            .with_context(|| format!("position: '{}', move: '{}'", self.position.fen(), mv))?;
        self.eval = None;
        self.positions_seen += 1;

        Ok(())
    }

    fn visit_comment(&mut self, comment: &[u8]) -> anyhow::Result<()> {
        let comment = str::from_utf8(comment)?;
        if comment == "book" {
            return Ok(());
        }

        if let Some(info) = comment.split('/').next() {
            if !info.starts_with("+M") && !info.starts_with("-M") {
                if let Ok(eval) = info.parse::<f64>() {
                    self.eval = Some((-eval * 100.0).round() as i16);
                }
            }
        }

        Ok(())
    }
}

impl GameVisitor {
    fn write(&mut self, eval: i16) -> anyhow::Result<()> {
        let sample = Sample {
            position: self.position.clone(),
            outcome: self
                .outcome
                .ok_or(anyhow::Error::msg("game has no outcome"))?,
            eval: Some(eval),
        }
        .pack()?;
        self.buffer.push(sample);
        self.positions_written += 1;
        Ok(())
    }

    fn take_buffer(&mut self) -> Vec<PackedSample> {
        mem::take(&mut self.buffer)
    }
}
