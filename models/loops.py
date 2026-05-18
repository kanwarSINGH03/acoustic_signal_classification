from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    cohen_kappa_score,
    auc,
)



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
    device: str = "mps",
):

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    if optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    device = torch.device(device)

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

        torch.save(model.state_dict(), model_path)
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

        torch.save(model.state_dict(), model_path)
    return history


def _logits_to_scores(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert model outputs to P(class=1 | x) scores.
    Supports:
      - Two-logit outputs: shape [B,2]
      - Single-logit outputs: shape [B] or [B,1]
    """
    if logits.ndim == 2 and logits.size(1) == 2:
        return F.softmax(logits, dim=1)[:, 1]
    elif logits.ndim == 2 and logits.size(1) == 1:
        return torch.sigmoid(logits[:, 0])
    elif logits.ndim == 1:
        return torch.sigmoid(logits)
    else:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

def _plot_threshold_views(idxs, trues, scores, threshold=0.5, class_names=("unstable (0)", "stable (1)")):
    idxs = np.asarray(idxs)
    trues = np.asarray(trues)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    mis = preds != trues

    # --- Per-class index vs score (two separate figures) ---
    for cls in (0, 1):
        mask = trues == cls
        if not mask.any():
            continue
        x = idxs[mask]
        s = scores[mask]
        p = preds[mask]
        mis_c = p != cls

        order = np.argsort(x)
        x, s, mis_c = x[order], s[order], mis_c[order]

        plt.figure(figsize=(12, 3.2))
        plt.plot(x, s, marker="o", ms=3, lw=0.8, color="orange")
        if mis_c.any():
            plt.scatter(x[mis_c], s[mis_c], s=24, facecolors="none", edgecolors="blue", linewidths=1.2, label="Misclassified")
        plt.hlines([threshold], xmin=x.min(), xmax=x.max(), linestyles="--", label=f"Threshold = {threshold:.2f}")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Test sample index (order from DataLoader)")
        plt.ylabel("Score = P(class=1 | x)")
        plt.title(f"Scores by Index — True = {class_names[cls]}")
        plt.legend(loc="best")
        plt.stable_layout()
        plt.show()

    # --- Combined histogram to see separation ---
    plt.figure(figsize=(8, 4))
    plt.hist(scores[trues == 0], bins=30, alpha=0.7, label=class_names[0])
    plt.hist(scores[trues == 1], bins=30, alpha=0.7, label=class_names[1])
    plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.2f}")
    plt.xlabel("Score = P(class=1 | x)")
    plt.ylabel("Count")
    plt.title("Score Distribution by True Class")
    plt.legend()
    plt.stable_layout()
    plt.show()

def test(
    model,
    model_path: str,
    test_loader,
    report: bool = False,
    score: bool = False,
    threshold_viz: bool = False,
    threshold: float = 0.5,
    device: str = "mps",
):
    """
    Load a trained model, evaluate on test_loader, always compute all metrics,
    and only display reports/plots when the corresponding flags are True.
    """
    print("[INFO] Testing the model")
    device = torch.device(device)

    # 1) Load model weights
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    # 2) Move to device and eval mode
    model.to(device)
    model.eval()

    # Accumulators
    correct = 0
    total = 0
    all_trues, all_scores, all_logits = [], [], []
    all_indices = []

    running_idx = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            scores1 = _logits_to_scores(logits)   # P(class=1 | x)
            preds = (scores1 >= threshold).long()

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_trues.append(y.cpu())
            all_scores.append(scores1.cpu())
            all_logits.append(logits.detach().cpu())

            idxs = torch.arange(running_idx, running_idx + y.size(0))
            all_indices.append(idxs)
            running_idx += y.size(0)

    # -----------------------------
    # Flatten everything
    # -----------------------------
    trues_arr = torch.cat(all_trues).numpy()
    scores_arr = torch.cat(all_scores).numpy()
    logits_all = torch.cat(all_logits)
    idxs_arr = torch.cat(all_indices).numpy()
    preds_arr = (scores_arr >= threshold).astype(int)

    # -----------------------------
    # Always compute all metrics
    # -----------------------------
    test_acc = correct / total if total > 0 else 0.0
    cm = confusion_matrix(trues_arr, preds_arr)

    fpr, tpr, roc_thresholds = roc_curve(trues_arr, scores_arr)
    roc_auc = auc(fpr, tpr)

    class_report = classification_report(
        trues_arr,
        preds_arr,
        target_names=["unstable (0)", "stable (1)"],
        zero_division=0,
    )

    cohen_kappa = cohen_kappa_score(trues_arr, preds_arr)

    print(f"Test accuracy (threshold={threshold:.2f}): {test_acc:.4f}")

    # -----------------------------
    # Display only when requested
    # -----------------------------
    if report:
        labels = ["unstable (0)", "stable (1)"]

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Reds", values_format=".0f")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

        print(f"\nClassification Report (threshold = {threshold:.2f}):\n")
        print(class_report)

    if threshold_viz:
        _plot_threshold_views(
            idxs_arr,
            trues_arr,
            scores_arr,
            threshold=threshold,
            class_names=("unstable (0)", "stable (1)"),
        )

    # -----------------------------
    # Return scores only if requested
    # -----------------------------
    if score:
        return {
            "accuracy": test_acc,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report": class_report,
            "y_true": trues_arr,
            "y_pred": preds_arr,
            "y_score": scores_arr,
            "logits": logits_all.numpy(),
            "indices": idxs_arr,
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
            "cohen_kappa": cohen_kappa,
        }

    return test_acc