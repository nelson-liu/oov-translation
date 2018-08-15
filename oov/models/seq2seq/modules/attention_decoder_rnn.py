import torch
import torch.nn as nn

from .attention import Attention
from .stacked_gru import StackedGRU
from .stacked_lstm import StackedLSTM
from ....utils.pytorch.general import get_sequence_mask_from_lengths


class AttentionDecoderRNN(nn.Module):
    def __init__(self, embed_dim, rnn_dim, vocab_size, padding_idx,
                 dropout=0.3, num_layers=2, input_feed=True, rnn_type="LSTM"):
        """
        Parameters
        ----------
        embed_dim: int
            The dimensionality of the embedding layer.

        rnn_dim: int
            The dimensionality of the decoder RNN hidden layer / output.
            Note that this should be the same rnn_dim used in the encoder.

        vocab_size: int
            The size of the vocabulary.

        padding_idx: int
            The index in the data that corresponds to padding.

        dropout: float (optional, default=0.1)
            The proportion of RNN outputs to dropout after each layer except
            the last.

        dropout: float, optional (default=0.3)
            The proportion of RNN outputs to drop out after each layer
            except the last.

        num_layers: int, optional (default=2)
            The number of layers to put in the RNN.

        input_feed: boolean, optional (default=True)
            Whether or not to append the attention vector to the embedding
            as input to the decoder RNN.

        rnn_type: str, optional (default="LSTM")
            One of [LSTM|GRU] to indicate the type of RNN to use.
        """
        super(AttentionDecoderRNN, self).__init__()
        self.embed_dim = embed_dim
        # Since the decoder is not bidirectional, hidden_dim = rnn_dim
        self.hidden_dim = self.rnn_dim = rnn_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.input_feed = input_feed
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim,
                                      padding_idx=self.padding_idx)
        input_size = (self.embed_dim + self.rnn_dim if self.input_feed else
                      self.embed_dim)
        if self.rnn_type == "LSTM":
            self.rnn = StackedLSTM(input_size, self.rnn_dim, dropout=dropout,
                                   num_layers=self.num_layers)
        elif self.rnn_type == "GRU":
            self.rnn = StackedGRU(input_size, self.rnn_dim, dropout=dropout,
                                  num_layers=self.num_layers)
        self.attn = Attention(self.rnn_dim)

    def forward(self, input_sequence, context, context_lengths,
                initial_hidden, initial_output):
        """
        Parameters
        ----------
        input_sequence: Variable(LongTensor)
            A Variable(LongTensor) of size (seq_len, batch_size) with the
            indices of the input words.

        context: Variable(FloatTensor)
            A Variable(FloatTensor) to attend to, generally the output of
            the encoder. Shape: (src_seq_len, batch_size, rnn_dim)

        context_lengths: List of int
            Encodes the context length for each batch. Shape: (batch_size)

        initial_hidden: Variable(FloatTensor)
            The initial hidden state to use in the decoder RNN.
            Shape: (num_layers, batch_size, rnn_dim)

        initial_output: (batch_size, hidden_dim)
            The initial output to use in the decoder RNN, since if input_feed
            is True we concat the RNN output with the embedded words.

        Returns
        -------
        outputs: Variable(FloatTensor)
            Represents the decoder output. Shape: (seq_len, batch_size, rnn_dim)

        hidden: Variable(FloatTensor)
            Represents the last hidden state of the decoder RNN.
            Shape: (num_layers, batch_size, rnn_dim)

        attention_weights: Variable(FloatTensor)
            The weight put on each element of the context. Shape: (batch_size, seq_len)
        """
        # Shape: (input_seq_len, batch_size, embed_dim)
        input_sequence_embedded = self.embedding(input_sequence)

        # Shape: (batch_size, max_seq_len)
        context_mask = get_sequence_mask_from_lengths(context_lengths)

        # Set the attention mask for the context
        self.attn.set_mask(context_mask)

        outputs = []
        hidden = initial_hidden
        output = initial_output
        # Iterate over the timesteps one at a time.
        for embedded_timestep in input_sequence_embedded.split(1):
            # Shape: (batch_size, embed_dim)
            embedded_timestep = embedded_timestep.squeeze(0)
            if self.input_feed:
                # Shape: (batch_size, embed_dim + rnn_dim)
                embedded_timestep = torch.cat([embedded_timestep, output], 1)

            # output shape: (batch_size, rnn_dim)
            # hidden shape: (num_layers, batch_size, rnn_dim)
            output, hidden = self.rnn(embedded_timestep, hidden)

            # output shape: (batch_size, rnn_dim)
            # attention_weights shape: (batch_size, seq_len)
            output, attention_weights = self.attn(output, context.transpose(0, 1))
            output = self.dropout(output)
            outputs += [output]

        # Shape: (seq_len, batch_size, rnn_dim)
        outputs = torch.stack(outputs)
        return outputs, hidden, attention_weights
