from torch.utils.data import DataLoader
import torch.nn as nn
import torch


def train(
    model,
    train_loader: DataLoader,
    batch_size: int,
    epochs: int,
    model_path: str,
    lr: float,
    weight_decay: float,
    optim: str,
):


    criterion = nn.CrossEntropyLoss()
    if optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    train_steps = len(train_loader.dataset) // batch_size

    for e in range(epochs):
        model.train()
        # initialize the total training and validation loss
        epoch_train_loss = 0

        train_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        mean_train_loss = epoch_train_loss / train_steps

        train_correct = train_correct / len(train_loader.dataset)
        history["train_loss"].append(mean_train_loss.cpu().detach().numpy())
        history["train_acc"].append(train_correct)
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print(
            "Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                mean_train_loss, train_correct
            )
        )

    torch.save(model, model_path)


def test(
    model_path: str, test_loader: DataLoader, report: bool = False, score: bool = False
):
    # test on the test set
    print("[INFO] Testing the model")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = torch.load(model_path, weights_only=False)
    model.to(device)
    test_correct = 0
    with torch.no_grad():
        model.eval()
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_acc = test_correct / len(test_loader.dataset)
    print(f"Test accuracy: {test_acc:.4f}")

    if report:
        return report
    if score:
        return score
