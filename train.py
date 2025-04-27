import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import time
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch
from model import TimeSeriesUNet
from data import create_dataset, Ddosattack
import pandas as pd
from torch import nn
import os
from torch.utils.data import DataLoader, TensorDataset

def combo_loss(predictions, targets, alpha=0.5, epsilon=1e-6):
    """
    predictions: Tensor of shape (batch_size, num_classes) with softmax probs
    targets: Tensor of shape (batch_size, num_classes) with one-hot labels
    """
    # Ensure predictions are float and bounded to avoid log(0)
    predictions = torch.clamp(predictions, epsilon, 1. - epsilon)
    targets = targets.float()

    # BCE Loss (for one-hot multi-class)
    bce = nn.BCELoss()(predictions, targets)

    # Dice Loss
    predictions_flat = predictions.view(-1)
    targets_flat = targets.view(-1)
    intersection = (predictions_flat * targets_flat).sum()
    dice = 1 - (2.0 * intersection + 1.0) / (
        predictions_flat.sum() + targets_flat.sum() + 1.0
    )

    # Combined loss
    return alpha * bce + (1 - alpha) * dice

print_interval = 100


def train_model(model, train_loader, test_loader, num_epochs, device, save_epoch = 20, learning_rate=0.001):
    """
    Train the time-series U-Net model and evaluate performance metrics.

    Parameters:
    model (nn.Module): The neural network model
    train_loader (DataLoader): DataLoader for training data
    test_loader (DataLoader): DataLoader for testing/validation data
    num_epochs (int): Number of training epochs
    learning_rate (float): Learning rate for optimizer
    device (str): Device to run training on ('cuda' or 'cpu')

    Returns:
    dict: Training history containing loss and metrics
    """

    # Move model to device
    model = model.to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize history dictionary to track metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # Initialize metrics for this epoch
        train_losses = []
        train_preds = []
        train_targets = []

        # Track time
        start_time = time.time()

        # Iterate over batches
        for i, (inputs, targets) in enumerate(train_loader):
            # print("inputs:", inputs.size())
            # print("target:", targets.size())
            # Move data to device
            inputs = inputs.to(device)

            # Permute inputs if needed [batch, seq_len, features] → [batch, features, seq_len]
            if inputs.shape[1] != model.encoder_blocks[0].conv[0].in_channels:
                inputs = inputs.permute(0, 2, 1)

            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            # print("output:", outputs.size())
            
            
            # Reshape targets if needed to match output shape
            if targets.dim() != outputs.dim():
                targets = targets.unsqueeze(1)

            # Calculate loss
            loss = combo_loss(outputs, targets.float())

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            train_losses.append(loss.item())

            # Convert outputs to binary predictions (threshold of 0.5)
            preds = (outputs > 0.5).float()

            # Store predictions and targets for metric calculation
            train_preds.append(preds.cpu().detach())
            train_targets.append(targets.cpu().detach())

            # Print progress every interval
            if (i + 1) % print_interval == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Iteration [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Time: {time.time() - start_time:.2f}s"
                )
                start_time = time.time()

        # Calculate epoch metrics for training
        train_loss = np.mean(train_losses)

        # Flatten predictions and targets for metric calculation
        train_preds_flat = torch.cat([p.flatten() for p in train_preds])
        train_targets_flat = torch.cat([p.flatten() for p in train_targets])

        train_preds_flat = torch.cat([p.flatten() for p in train_preds]).tolist()
        train_targets_flat = torch.cat([t.flatten() for t in train_targets]).tolist()

        train_acc = accuracy_score([t > 0.5 for t in train_targets_flat], [p > 0.5 for p in train_preds_flat])
        train_f1 = f1_score([t > 0.5 for t in train_targets_flat], [p > 0.5 for p in train_preds_flat], average="binary")

        # 保存训练指标
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)

        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                # Move data to device
                inputs = inputs.to(device)

                # Permute inputs if needed
                if inputs.shape[1] != model.encoder_blocks[0].conv[0].in_channels:
                    inputs = inputs.permute(0, 2, 1)

                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)
                print("outputs_size:", outputs.size())

                # Reshape targets if needed
                if targets.dim() != outputs.dim():
                    targets = targets.unsqueeze(1)

                # Calculate loss
                loss = combo_loss(outputs, targets.float())

                # Track loss
                val_losses.append(loss.item())

                # Convert outputs to binary predictions
                preds = (outputs > 0.5).float()

                # Store predictions and targets
                val_preds.append(preds.cpu())
                val_targets.append(targets.cpu())

        # Calculate epoch metrics for validation
        val_loss = np.mean(val_losses)

        # Flatten 验证集预测与标签
        val_preds_flat = torch.cat([p.flatten() for p in val_preds]).tolist()
        val_targets_flat = torch.cat([t.flatten() for t in val_targets]).tolist()

        val_acc = accuracy_score([t > 0.5 for t in val_targets_flat], [p > 0.5 for p in val_preds_flat])
        val_f1 = f1_score([t > 0.5 for t in val_targets_flat], [p > 0.5 for p in val_preds_flat], average="binary")

        # 保存验证指标
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Print epoch metrics
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(
            f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}"
        )
        print(
            f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}"
        )
        print("-" * 60)
        
        if epoch % save_epoch == 0:
            save_dir = "./checkpoints"
            os.makedirs(save_dir, exist_ok=True) 

            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            save_path = f"./checkpoints/model_epoch_{epoch+1}.pth"

            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    return history

def evaluate_model_detailed(model, test_loader, device, threshold=0.5):
    """
    Evaluates the model on test data and prints detailed confusion matrix metrics.

    Parameters:
    model (nn.Module): The trained neural network model
    test_loader (DataLoader): DataLoader containing test data
    device (str): Device to run evaluation on ('cuda' or 'cpu')
    threshold (float): Decision threshold for binary classification

    Returns:
    dict: Dictionary containing detailed metrics
    """

    # Set model to evaluation mode
    model.eval()
    model = model.to(device)

    # Initialize lists to store predictions and targets
    all_preds = []
    all_targets = []

    # No gradient computation needed for evaluation
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Permute inputs if needed [batch, seq_len, features] → [batch, features, seq_len]
            if inputs.shape[1] != model.encoder_blocks[0].conv[0].in_channels:
                inputs = inputs.permute(0, 2, 1)

            # Forward pass
            outputs = model(inputs)
            preds = outputs.argmax(dim=-1)     # [32, 16]
            targets_cls = targets.argmax(dim=-1)  # same as preds if one-hot or soft label

            all_preds.append(preds.cpu())
            all_targets.append(targets_cls.cpu())
    # Flatten predictions and targets to [B * T]
    all_preds_flat = torch.cat(all_preds).flatten().tolist()
    all_targets_flat = torch.cat(all_targets).flatten().tolist()

    # Compute overall accuracy
    accuracy = accuracy_score(all_targets_flat, all_preds_flat)

    # Confusion matrix and classification report (flattened over all time steps)
    conf_matrix = confusion_matrix(all_targets_flat, all_preds_flat)
    report = classification_report(all_targets_flat, all_preds_flat, digits=4)

    # Display metrics
    print("\n===== DETAILED EVALUATION METRICS =====")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)

    # Optional: Per-class F1, precision, recall, etc. can be extracted from report if needed
    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
    }

    return metrics

def preprocess_df(df):
    # 去除时间戳或包长为 0 的行
    df = df[(df["Timestamp"] != 0) & (df["Packet_Length"] != 0)].reset_index(drop=True)
    
    # 计算 IAT（Inter-Arrival Time）
    df["IAT"] = df["Timestamp"].diff().fillna(0)
    
    # 归一化 IAT（0 到 0.95 分位数之间）
    min_iat = 0
    max_iat = df["IAT"].quantile(0.95)
    df["IAT_norm"] = df["IAT"].clip(lower=min_iat, upper=max_iat) / max_iat

    # 归一化包长（最大按 1500 限制）
    df["PL_norm"] = df["Packet_Length"].clip(lower=0, upper=1500) / 1500

    # 保留必要字段
    df = df[["IAT_norm", "PL_norm", "Label"]].reset_index(drop=True)
    return df

def train_main():
    # Create model and dataloaders
    model = TimeSeriesUNet(input_features=2, base_filters=16, depth=2)

    train_test_ratio = 0.8
    segment_length = 16
    increment = 4
    batch_size = 32
    device = torch.device("cuda")

    df_amp = pd.read_csv(
        "./dataset/amp_dns.txt",
        delimiter=" ",
        header=None,
        names=["Timestamp", "Packet_Length", "Label"],
        dtype=float,
    )
    df_amp = preprocess_df(df_amp)
    dataset_amp_train, dataset_amp_test = create_dataset(
        df_amp, train_test_ratio, 1, segment_length, increment
    )

    # 读取其他数据集
    df_bruteforce = pd.read_csv(
        "./dataset/bruteforce_icmp.txt",
        delimiter=" ",
        header=None,
        names=["Timestamp", "Packet_Length", "Label"],
        dtype=float,
    )
    df_bruteforce = preprocess_df(df_bruteforce)
    dataset_bruteforce_train, dataset_bruteforce_test = create_dataset(
        df_bruteforce, train_test_ratio, 2, segment_length, increment
    )

    df_app = pd.read_csv(
        "./dataset/app_dns.txt",
        delimiter=" ",
        header=None,
        names=["Timestamp", "Packet_Length", "Label"],
        dtype=float,
    )
    df_app = preprocess_df(df_app)
    dataset_app_train, dataset_app_test = create_dataset(
        df_app, train_test_ratio, 3, segment_length, increment
    )

    df_whisper = pd.read_csv(
        "./dataset/Whisper_FUZZ.txt",
        delimiter=" ",
        header=None,
        names=["Timestamp", "Packet_Length", "Label"],
        dtype=float,
    )
    df_whisper = preprocess_df(df_whisper)
    dataset_whisper_train, dataset_whisper_test = create_dataset(
        df_whisper, train_test_ratio, 4, segment_length, increment
    )

    train_list = [dataset_amp_train, dataset_bruteforce_train, dataset_app_train,dataset_whisper_train]
    test_list = [dataset_amp_test, dataset_bruteforce_test, dataset_app_test,dataset_whisper_test]

    train_data = Ddosattack(train_list)
    test_data = Ddosattack(test_list)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Train model
    history = train_model(
        model = model,
        train_loader = train_loader,
        test_loader = test_loader,
        num_epochs = 100,
        device = device,
        learning_rate = 0.001,
    )

    metrics = evaluate_model_detailed(model, test_loader, device)

    # You can access specific metrics if needed
    print(f"F1 Score: {metrics['f1_score']}")

def test_main():
    # Create model and dataloaders
    model = TimeSeriesUNet(input_features=2, base_filters=16, depth=2)
    model_path = './checkpoints/model_epoch_81.pth'
    model.load_state_dict(torch.load(model_path, map_location='cuda'))

    train_test_ratio = 0.0
    segment_length = 16
    increment = 4
    batch_size = 32
    device = torch.device("cuda")
    
    df_amp = pd.read_csv(
        "./dataset/amp_ntp.txt",
        delimiter=" ",
        header=None,
        names=["Timestamp", "Packet_Length", "Label"],
        dtype=float,
    )
    df_amp = preprocess_df(df_amp)
    _, dataset_amp_test = create_dataset(
        df_amp, train_test_ratio, 1, segment_length, increment
    )

    test_list = [dataset_amp_test]
    test_data = Ddosattack(test_list)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    metrics = evaluate_model_detailed(model, test_loader, device)

