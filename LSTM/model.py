# Colin Brown, April 6, CS4442: AI 2 Final Report
# lstm_model.py Defines the LSTM model

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    lstm model for WR prediction.

    args:
        inputSize (int): num of input features.
        hiddenSize (int): num of features in the hidden state h.
        numLayers (int): num of recurrent layers (stacked lstm).
        outputSize (int): num of output values (1 for WR).
    """
    def __init__(self, inputSize, hiddenSize, numLayers, outputSize):
        super(LSTMModel, self).__init__() # inherit from nn.Module
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        # lstm layer; batch_first=True ensures (batch, seq, feature) tensor shape
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True)
        # fully connected layer maps lstm output to final prediction size
        self.fc = nn.Linear(hiddenSize, outputSize)


    def forward(self, x):
        """
        forward pass through the lstm model.
        Arg x (torch.Tensor): input tensor of shape (batch_size, sequence_length, input_size)
        returns torch.Tensor: output tensor of shape (batch_size, output_size)
        """
        # initialize hidden state (h0) and cell state (c0) with zeros
        h0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device) # shape: (num_layers, batch_size, hidden_size)
        c0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device) # same shape /\

        # pass input sequence and initial states through lstm layer
        out, _ = self.lstm(x, (h0, c0)) # lstm outputs features for each time step; we don't need the final hidden/cell states (_)

        # use output only from the last time step for prediction
        out = out[:, -1, :] # shape changes from (batch, seq, hidden) to (batch, hidden)

        # pass the last time step's output through the fully connected layer
        out = self.fc(out) # final linear layer projection
        return out