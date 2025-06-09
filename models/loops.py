from torch.utils.data import TensorDataset, DataLoader
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
        optimizer = torch.optim.Adam(
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

    nb_epochs = epochs
    # run for nb_epochs
    for e in range(nb_epochs):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        epoch_train_loss = 0
        #   epoch_val_loss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        train_correct = 0
        #   val_correct = 0

        for x, y in train_loader:

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            epoch_train_loss += loss
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        #   # switch off autograd for validation
        #   with torch.no_grad():
        #       # set the model in evaluation mode
        #       model.eval()
        #       # loop over the validation set
        #       for (x, y) in val_loader:

        #           x, y = x.to(device), y.to(device)
        #           pred = model(x)
        #           loss = criterion(pred, y)
        #           epoch_val_loss += loss
        #           val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # calculate the average epoch training and validation loss
        mean_train_loss = epoch_train_loss / train_steps
        #   mean_val_loss = epoch_val_loss / val_steps
        # calculate the training and validation accuracy
        train_correct = train_correct / len(train_loader.dataset)
        # val_correct = val_correct / len(val_loader.dataset)
        # update our training history
        history["train_loss"].append(mean_train_loss.cpu().detach().numpy())
        history["train_acc"].append(train_correct)
        #   history["val_loss"].append(mean_val_loss.cpu().detach().numpy())
        # history["val_acc"].append(val_correct)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, nb_epochs))
        print(
            "Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                mean_train_loss, train_correct
            )
        )
        #   print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
        #       mean_val_loss, val_correct))
        # save the model if the validation loss is less than the previous
        # if mean_val_loss - prev_mean_val_loss> 0.01:
        #   break
        # else:
        #   prev_mean_val_loss = mean_val_loss

    torch.save(model, model_path)


def test(
    model_path: str, test_loader: DataLoader, report: bool = False, score: bool = False
):
    # test on the test set
    print("[INFO] Testing the model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
