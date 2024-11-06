import os
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset


from lib.utils import load_empty_model_assests
from lib.cpdnet_util import SetArguments, CPDNetInit
from lib.cpdnet_datautil import DataUtil
from lib.lstmae_ensemble import AE_SkipLSTM_AR, AE_skipLSTM, AR


def model(args, params, data):
    # Initialize configurations for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpdnet_init, Data, cpdnet, cpdnet_tensorboard = load_empty_model_assests(
        params["ensemble_space"]
    )

    skip_sizes = params["skip_sizes"]
    for j in range(params["ensemble_space"]):
        skip = skip_sizes[j]
        args2 = SetArguments(
            data="data/" + args.dataset_name + ".txt",
            filename=f"./results_{[args.model_name , args.model_name , args.model_name][0]}/{args.dataset_name}/seed={args.seed}/",
            save="model",
            epochs=args.epochs,
            skip=skip,
            window=args.windows,
            batchsize=1,
            horizon=params["horizon"],
            highway=params["highway"],
            lr=params["lr"],
            GRUUnits=params["GRUUnits"],
            SkipGRUUnits=params["SkipGRUUnits"],
            debuglevel=50,
            optimizer="SGD",
            normalize=0,
            trainpercent=params["train_percent"],
            validpercent=0,
            no_validation=False,
            tensorboard="",
            predict="all",
            plot=True,
        )

        cpdnet_init[j] = CPDNetInit(args2, args_is_dictionary=True)
        cpdnet[j], cpdnet_tensorboard[j], Data[j] = offline_training(
            [args.model_name, args.model_name, args.model_name][j], j, cpdnet_init[j]
        )

    return cpdnet_init, Data, cpdnet, cpdnet_tensorboard


def offline_training(model_name, j, cpdnet_init):
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)

    # Dumping configuration
    cpdnet_init.dump()

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
        print("Training samples are low")
        exit(0)

    print(
        "Training shape: X:", str(Data.train[0].shape), " Y:", str(Data.train[1].shape)
    )
    print(
        "Validation shape: X:",
        str(Data.valid[0].shape),
        " Y:",
        str(Data.valid[1].shape),
    )
    print("Testing shape: X:", str(Data.test[0].shape), " Y:", str(Data.test[1].shape))

    if cpdnet_init.plot and cpdnet_init.autocorrelation is not None:
        AutoCorrelationPlot(Data[j], cpdnet_init)

    # Model selection and initialization
    if model_name == "AE_skipLSTM_AR":
        cpdnet = AE_SkipLSTM_AR(cpdnet_init, Data.train[0].shape)
        print(f"Number of parameters: {sum(p.numel() for p in cpdnet.parameters())}")
    elif model_name == "AE_skipLSTM":
        cpdnet = AE_skipLSTM(cpdnet_init, Data.train[0].shape)
    elif model_name == "AR":
        cpdnet = AR(cpdnet_init, Data.train[0].shape)

    if cpdnet is None:
        print("Model could not be loaded or created ... exiting!")
        exit(1)

    print("Compiling the model")

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
        print("Training model on normal data...")
        train(cpdnet, train_loader, valid_loader, optimizer, loss_fn, cpdnet_init)

    return cpdnet, None, Data


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
        print(f"Epoch [{epoch + 1}/{init.epochs}], Loss: {avg_train_loss:.4f}")

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
            print(f"Validation Loss: {avg_valid_loss:.4f}")

            # Log validation loss to TensorBoard if available
            if writer:
                writer.add_scalar("Validation Loss", avg_valid_loss, epoch)

    # Calculate and print total training time
    end_time = datetime.now()
    print("Total training time: ", str(end_time - start_time))

    # Close the TensorBoard writer
    if writer:
        writer.close()

    return model
