import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from lib.utils import load_empty_model_assests
from lib.cpdnet_util import CPDNetInit
from lib.cpdnet_datautil import DataUtil
from lib.lstmae_ensemble import AE_SkipLSTM_AR, AE_skipLSTM, AR


def model(args):

    cpdnet_init, Data, cpdnet = load_empty_model_assests(args["ensemble_space"])
    skip_sizes = args["skip_sizes"]
    for j in range(args["ensemble_space"]):
        skip = skip_sizes[j]
        cpdnet_init[j] = CPDNetInit(args, skip)
        cpdnet[j], Data[j] = offline_training(
            [args.model_name, args.model_name, args.model_name][j], cpdnet_init[j]
        )

    return (
        cpdnet_init,
        Data,
        cpdnet,
    )


def offline_training(model_name, cpdnet_init):
    # Reading data
    Data = DataUtil(
        cpdnet_init.data,
        cpdnet_init.trainpercent,
        cpdnet_init.validpercent,
        cpdnet_init.horizon,
        cpdnet_init.window,
        cpdnet_init.normalise,
    )

    if Data.train[0].shape[0] == 0:
        logging.info("Training samples are low")
        exit(0)
    logging.info(
        "Training shape: X: %s Y: %s", Data.train[0].shape, Data.train[1].shape
    )
    logging.info(
        "Validation shape: X: %s Y: %s", Data.valid[0].shape, Data.valid[1].shape
    )
    logging.info("Testing shape: X: %s Y: %s", Data.test[0].shape, Data.test[1].shape)

    # Model selection and initialization
    if model_name == "AE_skipLSTM_AR":
        cpdnet = AE_SkipLSTM_AR(cpdnet_init, Data.train[0].shape)
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in cpdnet.parameters())}"
        )
    elif model_name == "AE_skipLSTM":
        cpdnet = AE_skipLSTM(cpdnet_init, Data.train[0].shape)
    elif model_name == "AR":
        cpdnet = AR(cpdnet_init, Data.train[0].shape)

    if cpdnet is None:
        logging.info("Model could not be loaded or created ... exiting!")
        exit(1)

    logging.info("Compiling the model")

    # Move model to device (CPU or GPU)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    cpdnet.to(device)

    # Define optimizer
    optimizer = optim.SGD(cpdnet.parameters(), lr=cpdnet_init.lr)
    loss_fn = (
        nn.MSELoss()
    )  # Assuming MSE is used; replace with appropriate loss if different

    # DataLoader setup for training and validation

    # Ensure data is in torch-compatible dtype
    train_data = torch.tensor(Data.train[0], dtype=torch.float32)
    train_labels = torch.tensor(Data.train[1], dtype=torch.float32)

    # Use DataLoader with the converted data
    train_loader = DataLoader(
        TensorDataset(train_data, train_labels), batch_size=1, shuffle=True
    )

    valid_data = torch.tensor(Data.valid[0], dtype=torch.float32)
    valid_labels = torch.tensor(Data.valid[1], dtype=torch.float32)
    valid_loader = DataLoader(
        TensorDataset(valid_data, valid_labels), batch_size=1, shuffle=False
    )

    # Model training
    if cpdnet_init.train:
        logging.info("Training model on normal data...")
        train(cpdnet, train_loader, valid_loader, optimizer, loss_fn, cpdnet_init)

    return cpdnet, Data


from torch.utils.tensorboard import SummaryWriter
import torch


def train(
    model, train_loader, valid_loader, optimizer, loss_fn, init, tensorboard=None
):
    # Set up TensorBoard logging
    writer = SummaryWriter() if tensorboard else None

    # Move model to the appropriate device (CPU or GPU)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Track training start time
    start_time = datetime.now()

    # Training loop
    for epoch in range(init.epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize the running loss for averaging

        for inputs, targets in train_loader:
            # Move inputs and targets to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        # Calculate average loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        logging.info(f"Epoch [{epoch + 1}/{init.epochs}], Loss: {avg_train_loss:.4f}")

        # Log training loss to TensorBoard if available
        if writer:
            writer.add_scalar("Training Loss", avg_train_loss, epoch)

        # Validation step
        if valid_loader:
            model.eval()  # Set the model to evaluation mode
            valid_loss = 0.0

            with torch.no_grad():  # Disable gradient calculation for validation
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    valid_loss += loss.item()

            # Calculate average validation loss for this epoch
            avg_valid_loss = valid_loss / len(valid_loader)
            logging.info(f"Validation Loss: {avg_valid_loss:.4f}")

            # Log validation loss to TensorBoard if available
            if writer:
                writer.add_scalar("Validation Loss", avg_valid_loss, epoch)

    # Calculate and logging.info total training time
    end_time = datetime.now()
    logging.info("Total training time: %s", end_time - start_time)

    # Close the TensorBoard writer
    if writer:
        writer.close()

    return model


def eval_data(ensemble_space, cpdnet, x, y, device="cuda:7"):
    # Convert `x` and `y` to PyTorch tensors if they are numpy arrays
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32)

    # Move the tensors to the specified device
    x_device = x.to(device)
    y_device = y.to(device)

    # Initialize an empty list for losses
    loss = [None] * ensemble_space

    # Loop through each ensemble model
    for j in range(ensemble_space):
        model = cpdnet[j].to(device)  # Move model to the specified device

        # Set the model to evaluation mode
        model.eval()

        # Disable gradient computation for evaluation
        with torch.no_grad():
            predictions = model(x_device)  # Forward pass
            criterion = (
                torch.nn.MSELoss()
            )  # Use Mean Squared Error loss, modify if necessary
            loss[j] = criterion(predictions, y_device).item()  # Compute and store loss

    return np.asarray(loss)


def train2(
    x,
    y,
    ensemble_space,
    cpdnet,
    cpdnet_init,
    tensorboard=None,
    epochs=50,
    device="cuda:7",
):
    # Convert x and y to PyTorch tensors if they are numpy arrays
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32)

    # Move data to the specified device
    x_device = x.to(device)
    y_device = y.to(device)

    # Initialize tensorboard writers if specified
    writers = (
        [SummaryWriter() for _ in range(ensemble_space)]
        if tensorboard
        else [None] * ensemble_space
    )

    # Create dataset and dataloader for batching
    dataset = TensorDataset(x_device, y_device)

    for j in range(ensemble_space):
        model = cpdnet[j].to(device)  # Move the model to the specified device
        model.train()  # Set the model to training mode
        optimizer = Adam(
            model.parameters(), lr=cpdnet_init[j].lr
        )  # Initialize optimizer for each model

        # Create a DataLoader with the model's batch size
        dataloader = DataLoader(
            dataset, batch_size=cpdnet_init[j].batchsize, shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()  # Zero the gradients
                predictions = model(batch_x)  # Forward pass
                criterion = (
                    torch.nn.MSELoss()
                )  # Mean Squared Error loss, or change as needed
                loss = criterion(predictions, batch_y)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                total_loss += loss.item()

            # TensorBoard logging (if applicable)
            if writers[j]:
                writers[j].add_scalar("Loss/Train", total_loss / len(dataloader), epoch)

        # Close the writer after training
        if writers[j]:
            writers[j].close()

    return cpdnet  # Return updated models
