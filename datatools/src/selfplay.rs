use anyhow::Context;
use dama::{Color, Move, Outcome, Position, ToMove, UciMove};
use dataformat::{PackedSample, Sample};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, seq::IndexedRandom};
use std::{
    fmt::Write,
    path::PathBuf,
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::{
    fs::{File, OpenOptions},
    io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter},
    process::{self, Command},
    sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel},
};

use crate::shuffle::shuffle;

#[derive(clap::Args)]
pub struct Args {
    #[clap(short('o'), help("Output data file"))]
    output: PathBuf,
    #[clap(short('a'), long("append"))]
    append: bool,
    #[clap(short('c'), long("command"))]
    command: String,
    #[clap(long("games"))]
    games: u32,
    #[clap(long("concurrency"), default_value_t = 1)]
    concurrency: u32,
    #[clap(long("nodes"))]
    nodes: Option<u64>,
    #[clap(long("depth"))]
    depth: Option<u32>,
    #[clap(long("random-moves"))]
    random_moves: u32,
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    let mut output_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(!args.append)
        .append(args.append)
        .open(&args.output)
        .await
        .with_context(|| format!("failed to open output path `{}`", args.output.display()))?;

    let games_per_task = args.games / args.concurrency;
    let games_rem = args.games % args.concurrency;
    let (sample_send, sample_recv) = unbounded_channel();
    let (outcome_send, outcome_recv) = unbounded_channel();
    for n in 0..args.concurrency {
        let rounds = if n < games_rem {
            games_per_task + 1
        } else {
            games_per_task
        };
        let sample_send = sample_send.clone();
        let outcome_send = outcome_send.clone();
        let command = args.command.clone();
        tokio::spawn(run_games(
            sample_send,
            outcome_send,
            command,
            rounds,
            args.nodes,
            args.depth,
            args.random_moves,
        ));
    }
    drop(outcome_send);
    drop(sample_send);

    tokio::try_join!(
        show_progress(outcome_recv, args.games),
        write_to_file(sample_recv, &mut output_file),
    )?;

    shuffle(output_file, None).await?;

    Ok(())
}

async fn write_to_file(
    mut sample_recv: UnboundedReceiver<PackedSample>,
    output_file: &mut File,
) -> anyhow::Result<()> {
    let mut writer = BufWriter::new(output_file);
    let mut written = 0;
    while let Some(sample) = sample_recv.recv().await {
        writer.write_all(bytemuck::bytes_of(&sample)).await?;
        written += 1;
    }
    println!("{} positions written", written);
    writer.flush().await?;
    anyhow::Result::<()>::Ok(())
}

async fn show_progress(
    mut outcome_recv: UnboundedReceiver<Outcome>,
    games: u32,
) -> anyhow::Result<()> {
    let progress = ProgressBar::new(games as u64)
        .with_style(
            ProgressStyle::with_template(
                "\
            {spinner} [{elapsed_precise:.yellow}] [{bar:20}] \
            running games... {pos}/{len} games finished {msg}",
            )
            .unwrap()
            .progress_chars("##-"),
        )
        .with_message("| 0W - 0B - 0D");
    progress.enable_steady_tick(Duration::from_millis(50));

    let mut white_win = 0;
    let mut black_win = 0;
    let mut draw = 0;

    while let Some(outcome) = outcome_recv.recv().await {
        match outcome {
            Outcome::Winner(Color::White) => white_win += 1,
            Outcome::Winner(Color::Black) => black_win += 1,
            Outcome::Draw => draw += 1,
        }
        progress.inc(1);
        progress.set_message(format!("| {}W - {}B - {}D", white_win, black_win, draw));
    }
    progress.finish();

    anyhow::Result::<()>::Ok(())
}

async fn run_games(
    sample_sender: UnboundedSender<PackedSample>,
    outcome_sender: UnboundedSender<Outcome>,
    command: String,
    games: u32,
    nodes: Option<u64>,
    depth: Option<u32>,
    random_moves: u32,
) -> anyhow::Result<()> {
    let mut engine_white = Engine::new(
        Command::new(&command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?,
    )
    .await?;
    let mut engine_black = Engine::new(
        Command::new(&command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?,
    )
    .await?;

    for _ in 0..games {
        engine_white.new_game().await?;
        engine_black.new_game().await?;

        let position = random_opening(Position::new_initial(), random_moves, &mut rand::rng());

        let mut game = Game::from_position(position);
        let outcome = loop {
            if let Some(outcome) = game.outcome() {
                break outcome;
            }

            let engine = match game.position().side_to_move() {
                Color::White => &mut engine_white,
                Color::Black => &mut engine_black,
            };
            let (mv, eval) = engine.go(game.position(), Go { nodes, depth }).await?;
            game.play(&mv, eval);
        };
        outcome_sender.send(outcome)?;

        for (pos, mv, eval) in game.history() {
            if pos.is_in_check() || pos.is_capture(&mv) {
                continue;
            }

            if let Some(eval) = eval {
                let sample = Sample {
                    position: pos.clone(),
                    outcome,
                    eval: Some(eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16),
                }
                .pack()?;
                sample_sender.send(sample)?;
            }
        }
    }

    engine_white.quit().await?;
    engine_black.quit().await?;

    Ok(())
}

fn random_opening(start_position: Position, random_moves: u32, rng: &mut impl Rng) -> Position {
    'outer: loop {
        let mut position = start_position.clone();
        let mut ply = 2 * random_moves;
        if rng.random_bool(0.5) {
            ply -= 1;
        }

        for _ in 0..ply {
            let moves = position.legal_moves();
            if moves.is_empty() {
                continue 'outer;
            }
            let mv = moves.choose(rng).unwrap();
            position.play_unchecked(mv);
        }
        break position;
    }
}

struct Engine {
    stdin: process::ChildStdin,
    lines: io::Lines<BufReader<process::ChildStdout>>,
}

struct Go {
    nodes: Option<u64>,
    depth: Option<u32>,
}

impl Engine {
    async fn new(mut process: process::Child) -> anyhow::Result<Engine> {
        let stdin = process.stdin.take().expect("failed to get process stdin");
        let lines =
            BufReader::new(process.stdout.take().expect("failed to get process stdout")).lines();
        let mut engine = Engine { stdin, lines };
        engine.ping().await?;
        Ok(engine)
    }

    async fn ping(&mut self) -> anyhow::Result<()> {
        self.send("uci").await?;

        let start = Instant::now();
        let timeout = Duration::from_millis(5000);

        while let Some(cmd) = self.read().await? {
            if cmd.trim() == "uciok" {
                return Ok(());
            }
            if start.elapsed() > timeout {
                return Err(anyhow::Error::msg("engine response timeout"));
            }
        }

        Ok(())
    }

    async fn new_game(&mut self) -> anyhow::Result<()> {
        self.send("ucinewgame").await?;
        Ok(())
    }

    async fn quit(&mut self) -> anyhow::Result<()> {
        self.send("quit").await?;
        Ok(())
    }

    async fn go(&mut self, position: &Position, go: Go) -> anyhow::Result<(Move, Option<i32>)> {
        self.send(format!("position fen {}", position.fen()))
            .await?;
        let mut cmd = String::from("go");
        if let Some(depth) = go.depth {
            cmd.write_fmt(format_args!(" depth {}", depth))?;
        }
        if let Some(nodes) = go.nodes {
            cmd.write_fmt(format_args!(" nodes {}", nodes))?;
        }
        self.send(cmd).await?;

        let mut eval = None;
        while let Some(cmd) = self.read().await? {
            let mut parts = cmd.split_whitespace();
            match parts.next() {
                Some("bestmove") => {
                    let mv = parts.next().context("invalid 'bestmove' usage")?;
                    let mv = mv.parse::<UciMove>()?;
                    return Ok((mv.to_move(position)?, eval));
                }
                Some("info") => {
                    while let Some(part) = parts.next() {
                        if part == "score" {
                            match parts.next() {
                                Some("cp") => {
                                    let info_eval = parts
                                        .next()
                                        .context("centipawn score not present")?
                                        .parse::<i32>()?;
                                    if matches!(parts.next(), Some("upperbound" | "lowerbound")) {
                                        break;
                                    }
                                    eval = Some(info_eval);
                                    break;
                                }
                                _ => {
                                    eval = None;
                                    break;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Err(anyhow::Error::msg(
            "program finished before returning best move",
        ))
    }

    async fn send(&mut self, cmd: impl Into<String>) -> io::Result<()> {
        self.stdin
            .write_all(format!("{}\n", cmd.into()).as_bytes())
            .await?;
        Ok(())
    }

    async fn read(&mut self) -> io::Result<Option<String>> {
        self.lines.next_line().await
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    stack: Vec<Position>,
    data_stack: Vec<(Move, Option<i32>)>,
}

impl Game {
    #[inline]
    fn from_position(initial_position: Position) -> Self {
        Game {
            stack: vec![initial_position],
            data_stack: vec![],
        }
    }

    #[inline]
    fn position(&self) -> &Position {
        &self.stack[self.stack.len() - 1]
    }

    #[inline]
    fn play(&mut self, mv: &Move, eval: Option<i32>) {
        self.stack.push(self.position().clone());
        self.data_stack.push((*mv, eval));
        self.stack.last_mut().unwrap().play_unchecked(mv);
    }

    #[inline]
    fn history(&self) -> impl Iterator<Item = (&Position, Move, Option<i32>)> + '_ {
        self.stack
            .iter()
            .zip(self.data_stack.iter())
            .map(|(pos, &(mv, eval))| (pos, mv, eval))
    }

    #[inline]
    fn outcome(&self) -> Option<Outcome> {
        let moves = self.position().legal_moves();
        if moves.is_empty() {
            if self.position().is_in_check() {
                return Some(Outcome::Winner(!self.position().side_to_move()));
            } else {
                return Some(Outcome::Draw);
            }
        }
        if self.is_draw() {
            return Some(Outcome::Draw);
        }
        None
    }

    #[inline]
    fn is_draw(&self) -> bool {
        self.position().halfmove_clock() >= 100
            || self.repetitions() >= 3
            || self.position().is_insufficient_material()
    }

    #[inline]
    fn repetitions(&self) -> u32 {
        let mut repetitions = 0;
        let since_irreversible = self.position().halfmove_clock() as usize + 1;
        for position in self.stack.iter().rev().take(since_irreversible).step_by(4) {
            if position.hash() == self.position().hash() {
                repetitions += 1;
            }
        }
        repetitions
    }
}
