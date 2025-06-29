# teras_train

This repository contains most of the tools required for training simple NNUE chess networks. It was primarily created for training the Teras' networks, but with some adjustments, it can be used for any engine.

# Subcrates

- `dataformat/`: A small library for parsing and outputting the binary dataset format used by the trainer.
- `datatools/`: A binary utility tool used for creating and handling dataset files.
- `dataloader/`: Used for loading datasets into batches that can be used by the trainer.
- `train/`: Some python scripts responsible for training new networks.

# Limitations

It still only supports the very simple Board768 feature set, but support for other feature sets is coming soon.

# Acknowledgements

[nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch): This repository and specially [this document](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md) were the only reason I was able to write this tool.
