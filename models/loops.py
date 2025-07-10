from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def train(
    model,
    train_loader: DataLoader,
    val: bool,
    val_loader: DataLoader,
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
    if val:
        val_steps = len(val_loader.dataset) // batch_size

        for e in range(epochs):
            model.train()
            # initialize the total training and validation loss
            epoch_train_loss = 0
            epoch_val_loss = 0

            train_correct = 0
            val_correct = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss
                train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # switch off autograd for validation
            with torch.no_grad():
                # set the model in evaluation mode
                model.eval()
                # loop over the validation set
                for x, y in val_loader:
                    ###############################################################################
                    # TODO:                                                                       #
                    # 1. move x, y to device                                                      #
                    # 2. predict batch x, save in pred                                            #
                    # 3. update epoch_val_loss                                                    #
                    # 5. update val_correct                                                       #
                    ###############################################################################
                    # *****BEGIN YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    epoch_val_loss += loss
                    val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            mean_train_loss = epoch_train_loss / train_steps
            mean_val_loss = epoch_val_loss / val_steps
            # calculate the training and validation accuracy
            train_correct = train_correct / len(train_loader.dataset)
            val_correct = val_correct / len(val_loader.dataset)
            # update our training history
            history["train_loss"].append(mean_train_loss.cpu().detach().numpy())
            history["train_acc"].append(train_correct)
            history["val_loss"].append(mean_val_loss.cpu().detach().numpy())
            history["val_acc"].append(val_correct)
            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print(
                "Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                    mean_train_loss, train_correct
                )
            )
            print(
                "Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
                    mean_val_loss, val_correct
                )
            )

        torch.save(model, model_path)
    else:
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
    return history


import torch
from torch.utils.data import DataLoader


def test(
    model_path: str, test_loader: DataLoader, report: bool = False, score: bool = False
):
    """
    Load a trained model, evaluate on test_loader, print accuracy,
    and optionally display a confusion matrix / classification report.
    """
    print("[INFO] Testing the model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare model
    model = torch.load(model_path, weights_only=False)
    model.to(device)
    model.eval()

    # Metrics accumulators
    correct = 0
    total = 0
    all_preds = []
    all_trues = []

    # Inference loop
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            if report:
                all_preds.append(preds.cpu())
                all_trues.append(y.cpu())

    test_acc = correct / total
    print(f"Test accuracy: {test_acc:.4f}")

    if report:
        from sklearn.metrics import (
            confusion_matrix,
            classification_report,
            ConfusionMatrixDisplay,
        )

        preds_arr = torch.cat(all_preds).numpy()
        trues_arr = torch.cat(all_trues).numpy()

        cm = confusion_matrix(
            trues_arr,
            preds_arr,
        )
        ConfusionMatrixDisplay(cm, display_labels=["drummy (0)", "tight (1)"]).plot(
            cmap="Blues", values_format=".0f"
        )
        plt.show()

        print(
            "\nClassification Report:\n",
            classification_report(
                trues_arr,
                preds_arr,
                target_names=["drummy (0)", "tight (1)"],
                zero_division=0,
            ),
        )

    if score:
        return test_acc
