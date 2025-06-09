from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pytorch_lightning as pl
import model as m
import data

def open_dataloaders(train_path: str, val_path: str, batch_size: int, epoch_size: int, val_size: int) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(data.NnueDataset(train_path, batch_size, epoch_size), batch_size=None, sampler=None)
    val_loader = DataLoader(data.NnueDataset(val_path, batch_size, val_size), batch_size=None, sampler=None)
    return train_loader, val_loader

def main():
    parser = ArgumentParser(
        prog='TerasTrain', 
        description='A NNUE training utility for the Teras chess engine'
    )
    parser.add_argument('--dataset', type=str, help='Path to the dataset')
    parser.add_argument('--val-dataset', type=str, help='Path to the validation dataset')
    parser.add_argument('--name', type=str, help='Label for the output files')
    # parser.add_argument('--dump', type=str, default='.', help='Dump epoch models at specified path')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    # parser.add_argument('--lr-decay', type=float, default=0.98, help='Learning rate decay factor')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8192, help='Number of samples in each training batch')
    parser.add_argument('--epoch-size', type=int, default=1000000, help='Number of samples in each training epoch')
    parser.add_argument('--val-size', type=int, default=1000000, help='Number of validation samples')
    parser.add_argument('--eval-weight', type=float, default=0.0, help='0.0 to train on game results and 1.0 to train on engine evaluations, values in between interpolate between both')
    parser.add_argument('-o', "--output", type=str, help='Output file for the trained network')
    args = parser.parse_args()

    model = m.NNUE(lr=args.lr, eval_weight=args.eval_weight)
    trainer = pl.Trainer(max_epochs=args.epochs)
    train_data, val_data = open_dataloaders(args.dataset, args.val_dataset, args.batch_size, args.epoch_size, args.val_size)
    trainer.fit(model, train_data, val_data)
    model.write_to_file(args.output)

if __name__ == "__main__":
    main()
