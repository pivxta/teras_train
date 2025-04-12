use core::str;
use std::{fs::File, io::{BufReader, BufWriter, Write}, path::PathBuf, time::Duration};

use anyhow::Context;
use clap::Parser;
use dama::{pgn, Outcome, Position, SanMove};
use dataformat::Sample;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser)]
struct Options {
    #[clap(help("Input PGN files."))]
    inputs: Vec<PathBuf>,
    #[clap(short('o'), default_value("output.bin"))]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Options::parse();
    let input_files = args
        .inputs
        .iter()
        .map(|path| File::open(path)
            .with_context(|| format!("failed to open input file `{}`", path.display()))
            .map(|file| (path, file)))
        .collect::<Result<Vec<_>, _>>()?;

    let output_file = File::create(&args.output)
        .with_context(|| format!("failed to open output path `{}`", args.output.display()))?;

    let mut writer = BufWriter::new(&output_file);
    let mut visitor = GameVisitor {
        writer: &mut writer,
        position: Position::new_initial(),
        outcome: None,
        skip: false,
        games_skipped: 0,
        games_read: 0,
        positions_written: 0,
        positions_seen: 0
    };

    for (input_path, input_file) in input_files {
        let mut reader = pgn::Reader::new(BufReader::new(input_file));        
        let progress = ProgressBar::new_spinner()
            .with_message(format!("reading games from `{}`", input_path.display()))
            .with_style(ProgressStyle::with_template("{spinner} [{elapsed_precise:.yellow}] {msg}: {human_pos} games read").unwrap());
        progress.enable_steady_tick(Duration::from_millis(100));
        loop {
            match reader.visit_game(&mut visitor) {
                Ok(true) => {},
                Ok(false) => break,
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
        progress.finish();
    }

    if visitor.positions_written > 0 {
        eprintln!("Done.");
        eprintln!("{} games read", visitor.games_read);
        eprintln!("{} games skipped", visitor.games_skipped);
        eprintln!("{} positions written", visitor.positions_written);
        eprintln!("{} positions seen", visitor.positions_seen);
    }

    Ok(())
}

struct GameVisitor<W> {
    writer: W,
    skip: bool,
    position: Position,
    outcome: Option<Outcome>,

    positions_written: u32,
    positions_seen: u32,
    games_read: u32,
    games_skipped: u32,
}

impl<W> pgn::Visitor for GameVisitor<W> 
where 
    W: Write
{
    type Error = anyhow::Error;

    fn prepare(&mut self) {
        self.position = Position::new_initial(); 
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

    fn visit_move(
        &mut self,
        _number: Option<u32>,
        mv: SanMove,
    ) -> anyhow::Result<()> {
        self.position.play(&mv)
            .with_context(|| format!("position: '{}', move: '{}'", self.position.fen(), mv))?;
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
                let eval = if let Ok(eval) = info.parse::<f64>() {
                    (-eval * 100.0).round() as i16
                } else {
                    return Ok(())
                };

                self.write(eval)?;
            }
        }

        Ok(()) 
    }
}

impl<W> GameVisitor<W>
where
    W: Write
{
    fn write(&mut self, eval: i16) -> anyhow::Result<()> {
        let sample = Sample {
            position: self.position.clone(),
            outcome: self.outcome.ok_or(anyhow::Error::msg("game has no outcome"))?,
            eval: Some(eval)
        }.pack()?;
        bincode::encode_into_std_write(&sample, &mut self.writer, bincode::config::standard())?;
        self.positions_written += 1;
        Ok(())
    }
}
