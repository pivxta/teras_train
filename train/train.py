from data import DataLoader
from argparse import ArgumentParser
import model as m
import torch
import sys

BATCHES_PER_LOG = 200 

class WeightClipper:
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, "weight"):
            weights = module.weight.data
            weights = weights.clamp(-1.98, 1.98)
            module.weight.data = weights

def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    loader: DataLoader, 
    *, 
    name: str,
    output_path: str,
    lr_decay: float,
    epochs: int, 
    eval_scale: float,
    eval_factor: float
):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    loss_fn = torch.nn.BCELoss()
    batches_since_log = 0 
    loss_since_log = torch.zeros((1,))

    for epoch in range(epochs):
        print(f"starting epoch {epoch}, with learning rate of {optimizer.param_groups[0]['lr']}")

        epoch_loss = torch.zeros((1,))
        epoch_batches = 0

        for n, batch in enumerate(loader):
            optimizer.zero_grad()
            prediction = model(batch)
            target = (eval_factor * torch.sigmoid(batch.evals * eval_scale) 
                + (1.0 - eval_factor) * batch.outcomes)
            loss = loss_fn(prediction, target) 
            loss.backward()
            optimizer.step()
            model.apply(WeightClipper())

            with torch.no_grad():
                epoch_loss += loss
                loss_since_log += loss

            epoch_batches += 1
            batches_since_log += 1
            if n == 0 or batches_since_log > BATCHES_PER_LOG:
                mean_loss = loss_since_log.item() / batches_since_log
                print(f"epoch: {epoch}, positions: {n * batch.size}, loss: {mean_loss}")
                loss_since_log = torch.zeros((1,))
                batches_since_log = 0

        scheduler.step()

        print(f"finished epoch {epoch}")
        print(f"average loss: {epoch_loss.item() / epoch_batches}")

        torch.save(model.state_dict(), f"models/{name}_epoch{epoch:04}.nn")


def main():
    parser = ArgumentParser(
        prog='TerasTrain', 
        description='A NNUE training utility for the Teras chess engine'
    )
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--name', type=str, help='Label for the output files')
    parser.add_argument('--output-path', type=str, default='.', help='Label for the output files')
    parser.add_argument('--initial-lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='Learning rate decay factor')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8192, help='Number of samples in each training batch')
    parser.add_argument('--eval-weight', type=float, default=0.0, help='Importance of the engine evaluation in training')
    parser.add_argument('--eval-scale', type=float, default=400.0, help='Scaling factor for engine evaluations')
    args = parser.parse_args()

    model = m.TerasNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
    with DataLoader(args.dataset_path, batch_size=args.batch_size) as loader:
        train(
            model, 
            optimizer, 
            loader, 
            name=args.name,
            output_path=args.output_path,
            lr_decay=args.lr_decay,
            epochs=args.epochs, 
            eval_factor=args.eval_weight, 
            eval_scale=1.0/args.eval_scale
        )

if __name__ == "__main__":
    main()
