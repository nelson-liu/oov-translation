import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from ....utils.pytorch.general import get_rnn_class


class EncoderRNN(nn.Module):
    def __init__(self, embed_dim, rnn_dim, vocab_size, padding_idx,
                 dropout=0.3, num_layers=2, bidirectional=True,
                 rnn_type="LSTM"):
        """Create a RNN for encoding sequences.

        Parameters
        ----------
        embed_dim: int
            The dimensionality of the embedding layer.

        rnn_dim: int
            The dimensionality of the RNN outputs. If bidirectional is True
            the rnn hidden_dim will be rnn_dim / 2, else it will be rnn_dim.

        vocab_size: int
            The size of the vocabulary.

        padding_idx: int
            The index in the data that corresponds to padding.

        dropout: float, optional (default=0.3)
            The proportion of RNN outputs to drop out after each layer
            except the last.

        num_layers: int, optional (default=2)
            The number of layers to put in the RNN.

        bidirectional: boolean, optional (default=True)
            Whether to make the encoder bidirectional or not.

        rnn_type: str, optional (default="LSTM")
            One of [LSTM|GRU] to indicate the type of RNN to use.
        """
        super(EncoderRNN, self).__init__()
        self.embed_dim = embed_dim
        self.rnn_dim = rnn_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        self.num_directions = 2 if self.bidirectional else 1
        assert self.rnn_dim % self.num_directions == 0

        # Note that hidden_dim refers to the size of the
        # last dimension of the hidden state and rnn_dim
        # refers to the size of the last dimension of the output.
        # They are the same if the rnn is not bidirectional
        self.hidden_dim = self.rnn_dim // self.num_directions

        self.rnn = get_rnn_class(self.rnn_type)(
            self.embed_dim, self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim,
                                      padding_idx=self.padding_idx)

    def forward(self, input_sequence, input_lengths, hidden=None):
        """
        This model takes in an input sequence of shape
        (batch_size, seq_len), runs it through an embedding layer,
        and then encodes the embedded sequence with a RNN.

        Parameters
        ----------
        input_sequence: Variable(LongTensor)
            A Variable(LongTensor) of shape (seq_len, batch_size)
            representing a batch of sequences. The sequences must
            be sorted in decreasing order of length.

        input_lengths: LongTensor
            A LongTensor of shape (batch_size) indicating the list of each
            word (not including padding) in input_words. The list should be
            sorted in decreasing order, since the input_words should be
            sorted in decreasing order of length.

        Returns
        -------
        outputs: Variable(FloatTensor)
            A Variable(FloatTensor) of shape (seq_len, batch_size, rnn_dim)
            representing the outputs of the EncoderRNN.

        last_hidden: Variable(FloatTensor)
            A Variable(FloatTensor) of shape (num_layers * num_directions,
            batch_size, hidden_dim) representing the last hidden state.
        """
        # Shape: (batch_size, seq_len, embed_dim)
        input_sequence_embedded = self.embedding(input_sequence)
        # Shape: List of length batch_size
        input_lengths_list = input_lengths.view(-1).tolist()

        # Pack the sequence to run it through the RNN, since it is variable length.
        input_sequence_packed = pack(input_sequence_embedded,
                                     input_lengths_list)

        # last_hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        packed_outputs, last_hidden = self.rnn(input_sequence_packed)

        # output shape: (seq_len, batch_size, rnn_dim)
        # lengths shape: (batch_size)
        outputs, lengths = unpack(packed_outputs)
        return outputs, last_hidden, lengths
