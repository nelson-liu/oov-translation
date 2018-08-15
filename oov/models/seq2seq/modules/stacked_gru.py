import torch
import torch.nn as nn


class StackedGRU(nn.Module):
    """
    A GRU with its cells stacked upon each other, for use in the decoder.
    """
    def __init__(self, input_size, rnn_size, dropout=0.3, num_layers=2):
        super(StackedGRU, self).__init__()
        # Instantiate layer instead of using functional API for dropout so
        # it shows up in the StackedGRU string representation.
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        """
        Encodes one timestep of input with the StackedGRU. For example, if
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
            The output of the GRU for this timestep. Shape: (batch_size, rnn_dim)
        hidden: Variable(FloatTensor)
            The hidden states of the StackedGRU layers for this timestep.
            Shape: (num_layers, batch_size, rnn_dim)
        """
        hidden_states = []
        for i, layer in enumerate(self.layers):
            # Get the hidden state from the GRUCell. Shape: (batch_size, rnn_dim)
            layer_hidden_state = layer(input, hidden[i])

            # Input to next layer is hidden state
            input = layer_hidden_state

            # Dropout after all layers except the last.
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            hidden_states += [layer_hidden_state]

        hidden_states = torch.stack(hidden_states)
        return input, hidden_states
