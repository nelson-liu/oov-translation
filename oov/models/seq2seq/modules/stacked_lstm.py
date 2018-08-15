import torch
import torch.nn as nn


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, input_size, rnn_size, dropout=0.3, num_layers=2):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        """
        Encodes one timestep of input with the StackedLSTM. For example, if
        we are encoding an embedded sequence of shape (seq_len, batch_size, embed_dim)
        then this class encodes successive slices of shape (batch_size, input_size)
         where there are seq_len of them in total.

        Parameters
        ----------
        input: Variable(FloatTensor)
            Contains the input features. Shape: (batch_size, input_size)

        hidden: Variable(FloatTensor)
            The hidden states to use in each layer of the stacked GRU.
            Shape: (num_layers, batch_size, rnn_dimension)

        Returns
        -------
        output: Variable(FloatTensor)
            The output of the LSTM for this timestep. Shape: (batch_size, rnn_dim)
        hidden: tuple(Variable(FloatTensor), Variable(FloatTensor))
            A tuple of (hidden states, cell states) of the StackedLSTM
            layers for this timestep. Shape: ((num_layers, batch_size,
            rnn_dim), (num_layers, batch_size, rnn_dim))
        """
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)
