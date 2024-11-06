import torch
from torch import nn


class DropoutRNNCellMixin:
    def __init__(self, dropout=0.0, recurrent_dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def reset_dropout_mask(self):
        self._dropout_mask = None

    def reset_recurrent_dropout_mask(self):
        self._recurrent_dropout_mask = None

    def get_dropout_mask_for_cell(self, inputs, training=True):
        if self.dropout > 0 and training:
            if self._dropout_mask is None:
                self._dropout_mask = torch.bernoulli(
                    (1 - self.dropout) * torch.ones_like(inputs)
                )
            return self._dropout_mask
        return None

    def get_recurrent_dropout_mask_for_cell(self, inputs, training=True):
        if self.recurrent_dropout > 0 and training:
            if self._recurrent_dropout_mask is None:
                self._recurrent_dropout_mask = torch.bernoulli(
                    (1 - self.recurrent_dropout) * torch.ones_like(inputs)
                )
            return self._recurrent_dropout_mask
        return None


class PreARTrans(nn.Module):
    def __init__(self, hw):
        """
        hw: Number of timeseries values to consider for the linear layer (AR layer)
        """
        super(PreARTrans, self).__init__()
        self.hw = hw

    def forward(self, x):
        """
        Forward pass of the PreARTrans layer.

        Arguments:
        x -- Input tensor with shape (batch_size, sequence_length, num_series)

        Returns:
        output -- Reshaped tensor for AR layer processing
        """
        # Get the batch size and input shape
        batchsize = x.size(0)
        num_series = x.size(2)

        # Select only 'highway' length of input from the end of sequence dimension
        output = x[:, -self.hw :, :]

        # Permute to place the time-series dimension in the middle
        output = output.permute(0, 2, 1)

        # Reshape for batch processing, combining batch and time-series dimensions
        output = output.reshape(batchsize * num_series, self.hw)

        return output


class PostARTrans(nn.Module):
    def __init__(self, m):
        """
        m: Number of timeseries
        """
        super(PostARTrans, self).__init__()
        self.m = m

    def forward(self, x, original_model_input):
        """
        Forward pass of the PostARTrans layer.

        Arguments:
        x -- Output of the Dense(1) layer, shape (batch_size * num_series, 1)
        original_model_input -- Original input tensor to the model for batch size reference

        Returns:
        output -- Reshaped tensor with batch size restored
        """
        # Get the batch size from the original input
        batchsize = original_model_input.size(0)

        # Reshape to have batchsize and num_series
        output = x.view(batchsize, self.m)

        return output


import torch
import torch.nn as nn


import torch
import torch.nn as nn


import torch
import torch.nn as nn


class skip_LSTMCell(nn.Module):
    def __init__(
        self,
        units,
        input_dim,
        skip,
        dropout=0.0,
        recurrent_dropout=0.0,
        device="cuda:7",
    ):
        super(skip_LSTMCell, self).__init__()
        self.units = units
        self.input_dim = input_dim[-1]
        self.skip = skip
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.device = device  # Track the device for consistency
        self.state_size = (units, units, 1, units * skip)

        # LSTM weights
        self.kernel = nn.Parameter(torch.Tensor(self.input_dim, 4 * units)).to(
            self.device
        )
        self.recurrent_kernel = nn.Parameter(torch.Tensor(units, 4 * units)).to(
            self.device
        )
        self.kernel2 = nn.Parameter(torch.Tensor(units, units)).to(self.device)
        self.bias = nn.Parameter(torch.zeros(4 * units)).to(self.device)
        self.bias2 = nn.Parameter(torch.zeros(units)).to(self.device)

        # Skip connection parameters
        self.s0 = nn.Parameter(torch.tensor(0.5)).to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        nn.init.xavier_uniform_(self.kernel2)

    def forward(self, inputs, states):
        # Unpack states
        h_tm1, c_tm1, step, prev_h = states

        # Check h_tm1 shape
        if h_tm1.size(1) != 4 * self.units:
            raise ValueError(
                f"h_tm1 should have shape (batch_size, {4 * self.units}), but got {h_tm1.size()}"
            )

        # Split h_tm1 into four parts for the LSTM gates
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1.chunk(4, dim=1)

        # Compute gates
        x = inputs.mm(self.kernel) + self.bias  # Shape: (batch_size, 4 * units)
        x_i, x_f, x_c, x_o = torch.split(x, self.units, dim=1)

        # Perform matrix multiplication with the split parts
        i = torch.sigmoid(x_i + h_tm1_i.mm(self.recurrent_kernel[:, : self.units]))
        f = torch.sigmoid(
            x_f + h_tm1_f.mm(self.recurrent_kernel[:, self.units : self.units * 2])
        )
        c = f * c_tm1 + i * torch.tanh(
            x_c + h_tm1_c.mm(self.recurrent_kernel[:, self.units * 2 : self.units * 3])
        )
        o = torch.sigmoid(x_o + h_tm1_o.mm(self.recurrent_kernel[:, self.units * 3 :]))
        h = o * torch.tanh(c)

        # Skip connection
        new_h_skip = torch.sigmoid(
            prev_h[:, : self.units].mm(self.kernel2) + self.bias2
        )
        h = self.s0 * h + new_h_skip * (1 - self.s0)

        # Update h_tm1 with the new values for the next timestep
        h_tm1 = torch.cat([i, f, c, o], dim=1)  # Rebuild h_tm1 with the four parts

        # Update previous hidden state
        prev_h = torch.cat(
            (prev_h[:, self.units :], h), dim=1
        )  # Shape: (batch_size, units * skip)

        return h, [h_tm1, c, step + 1, prev_h]


class AE_SkipLSTM_AR(nn.Module):
    def __init__(self, init, input_shape):
        super(AE_SkipLSTM_AR, self).__init__()

        self.m = input_shape[3]
        self.tensor_shape = input_shape[1:]

        # Initialize skip_LSTMCell
        if init.skip > 0:
            self.encoder_cell = skip_LSTMCell(init.SkipGRUUnits, input_shape, init.skip)
            self.decoder_cell = skip_LSTMCell(
                init.SkipGRUUnits, [init.SkipGRUUnits], init.skip
            )

        # Dense layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(
            init.SkipGRUUnits * input_shape[2], input_shape[2] * self.m
        )
        self.reshape = lambda x: x.view(-1, input_shape[2], self.m)

        # Autoregressive layers
        if init.highway > 0:
            self.highway_layers = nn.ModuleList(
                [PreARTrans(init.highway) for _ in range(input_shape[1] - 1)]
            )
            self.final_dense = nn.Linear(input_shape[2], 1)
            self.post_ar_trans = PostARTrans(self.m)

    def custom_rnn(self, cell, inputs):
        batch_size = inputs.size(0)
        units = cell.units

        # Initialize hidden states with the correct shape
        h_tm1 = torch.zeros(batch_size, 4 * units).to(
            cell.device
        )  # Hidden state with correct shape
        c_tm1 = torch.zeros(batch_size, units).to(cell.device)  # Cell state
        step = torch.zeros(batch_size, 1).to(cell.device)  # Step counter
        prev_h = torch.zeros(batch_size, units * cell.skip).to(
            cell.device
        )  # Skip connection

        outputs = []
        for t in range(inputs.size(1)):
            output, (h_tm1, c_tm1, step, prev_h) = cell(
                inputs[:, t], [h_tm1, c_tm1, step, prev_h]
            )
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1), h_tm1

    def forward(self, X):
        X1 = X[:, 0, :, :]
        X2 = X[:, 1:, :, :]

        # Encoder
        if hasattr(self, "encoder_cell"):
            SE, _ = self.custom_rnn(self.encoder_cell, X1)
            RE = SE

        P = RE

        # Decoder
        if hasattr(self, "decoder_cell"):
            SD, _ = self.custom_rnn(self.decoder_cell, P)
            RD = SD

        # Dense layer for reshaping the decoder output
        RD = self.flatten(RD)
        RD = self.dense(RD)
        Y = self.reshape(RD)

        # Autoregressive component
        if hasattr(self, "highway_layers"):
            Z2 = []
            for i in range(X2.size(1)):
                Z = self.highway_layers[i](X2[:, i, :, :])
                Z = self.flatten(Z)
                Z = self.final_dense(Z)
                Z = self.post_ar_trans(Z, X)
                Z2.append(Z.unsqueeze(1))

            Z2 = torch.cat(Z2, dim=1)
            Y = Y + Z2

        return Y


class AE_skipLSTM(nn.Module):
    def __init__(self, init, input_shape):
        super(AE_skipLSTM, self).__init__()

        # m is the number of time-series
        self.m = input_shape[3]
        self.sequence_length = input_shape[2]

        # Initialize skip_LSTMCell
        if init.skip > 0:
            self.encoder_cell = skip_LSTMCell(init.SkipGRUUnits, init.skip)
            self.decoder_cell = skip_LSTMCell(init.SkipGRUUnits, init.skip)

        # Dense layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(
            self.sequence_length * self.m, self.sequence_length * self.m
        )
        self.reshape = lambda x: x.view(-1, self.sequence_length, self.m)

    def custom_rnn(self, cell, inputs, return_sequences=False):
        # Custom RNN loop for skip_LSTMCell
        batch_size = inputs.size(0)
        hidden_state, carry_state = torch.zeros(batch_size, cell.units), torch.zeros(
            batch_size, cell.units
        )
        step = torch.zeros(batch_size, 1)
        prev_h = torch.zeros(batch_size, cell.units * cell.skip)
        outputs = []

        for t in range(inputs.size(1)):
            output, (hidden_state, carry_state, step, prev_h) = cell(
                inputs[:, t], [hidden_state, carry_state, step, prev_h]
            )
            if return_sequences:
                outputs.append(output.unsqueeze(1))

        if return_sequences:
            return torch.cat(outputs, dim=1), hidden_state
        else:
            return output, hidden_state

    def forward(self, X):
        X1 = X[:, 0, :, :]  # Slice the first timestep
        X2 = X[:, 1:, :, :]  # Slice the remaining timesteps

        """------------------------  Encoder   --------------------------"""
        # Pass X1 through the encoder
        if hasattr(self, "encoder_cell"):
            RE, _ = self.custom_rnn(self.encoder_cell, X1)

        # Repeat vector
        P = RE.unsqueeze(1).repeat(1, self.sequence_length, 1)

        """------------------------  Decoder   --------------------------"""
        # Pass repeated encoding through decoder
        if hasattr(self, "decoder_cell"):
            RD, _ = self.custom_rnn(self.decoder_cell, P, return_sequences=True)

        # Dense layer
        RD = self.flatten(RD)
        RD = self.dense(RD)
        Y = self.reshape(RD)

        return Y


class AR(nn.Module):
    def __init__(self, init, input_shape):
        super(AR, self).__init__()

        # m is the number of time-series
        self.m = input_shape[3]
        self.highway = init.highway
        self.sequence_length = input_shape[2]

        # Define the PreARTrans and PostARTrans layers for each timestep
        self.pre_ar_layers = nn.ModuleList(
            [PreARTrans(self.highway) for _ in range(self.sequence_length - 1)]
        )
        self.post_ar_layer = PostARTrans(self.m)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(self.highway, 1)

    def forward(self, X):
        # Split X into X1 and X2 along the timestep dimension
        X1 = X[:, 0, :, :]  # First timestep
        X2 = X[:, 1:, :, :]  # Remaining timesteps

        # Initialize an empty list to collect outputs
        Z2 = []

        for i in range(X2.size(1)):
            Z = self.pre_ar_layers[i](
                X2[:, i, :, :]
            )  # Apply PreARTrans for the current timestep
            Z = self.flatten(Z)
            Z = self.dense(Z)
            Z = self.post_ar_layer([Z, X])  # PostARTrans with Z and original input X
            Z = Z.unsqueeze(1)  # Add timestep dimension

            Z2.append(Z)

        # Concatenate along the sequence dimension to match the original behavior
        Y = torch.cat(Z2, dim=1)

        return Y
