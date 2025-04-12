from data import DataLoader
import model as m
import torch

BATCHES_PER_LOG = 200 

def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    loader: DataLoader, 
    *, 
    epochs: int, 
    eval_scale: float,
    eval_factor: float
):
    loss_fn = torch.nn.BCELoss()
    batches_since_log = 0 
    loss_since_log = torch.zeros((1,))

    for epoch in range(epochs):
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
        
        print(f"finished epoch {epoch}")
        print(f"average loss: {epoch_loss.item() / epoch_batches}")

        torch.save(model.state_dict(), f"{epoch:04}.nn")


def main():
    model = m.TerasNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 3
    with DataLoader("output.bin",batch_size=8192) as loader:
        train(model, optimizer, loader, epochs=3, eval_factor=0.0, eval_scale=1.0/400.0)

if __name__ == "__main__":
    main()
