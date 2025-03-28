use std::{fs::File, io::{BufReader, BufWriter}, path::PathBuf};

use anyhow::Context;
use clap::Parser;
use dama::pgn;
use tempfile::NamedTempFile;

#[derive(Parser)]
struct Options {
    #[clap(help("Input PGN files."))]
    inputs: Vec<PathBuf>,
    #[clap(short('o'))]
    output: Vec<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Options::parse();
    let input_files = args
        .inputs
        .iter()
        .map(|path| File::open(path)
            .with_context(|| format!("failed to open input file `{}`", path.display())))
        .collect::<Result<Vec<_>, _>>()?;
    let output_file = NamedTempFile::new().with_context(|| format!("failed to create temporary output file"))?;
    let mut writer = BufWriter::new(output_file);

    for input_file in input_files {
        let reader = pgn::Reader::new(BufReader::new(input_file));
        
    }

    Ok(())
}

struct GameVisitor<W> {
    writer: W,

}

impl pgn::Visitor for GameVisitor {
    fn prepare(&mut self) {
        
    }
}