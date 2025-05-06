mod extract;
mod merge;
mod selfplay;
mod shuffle;
use clap::{Parser, Subcommand};

#[derive(Subcommand)]
enum Command {
    #[clap(about("Extracts positions from labeled PGN files"))]
    Extract(extract::Args),
    #[clap(about("Shuffles data files"))]
    Shuffle(shuffle::Args),
    #[clap(about(
        "Runs games with the specified UCI compliant engine, outputting the resulting data to a file"
    ))]
    Selfplay(selfplay::Args),
    #[clap(about("Merges two or more data files"))]
    Merge(merge::Args),
}

#[derive(Parser)]
#[command(version, about)]
struct Options {
    #[command(subcommand)]
    command: Command,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let options = Options::parse();
    match options.command {
        Command::Extract(args) => extract::run(args).await?,
        Command::Shuffle(args) => shuffle::run(args).await?,
        Command::Selfplay(args) => selfplay::run(args).await?,
        Command::Merge(args) => merge::run(args).await?,
    }
    Ok(())
}
