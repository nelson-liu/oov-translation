import torch
from torch.autograd import Variable
import torch.nn as nn


class Seq2Seq(nn.Module):
    """
    A container class to hold the encoder and decoder and a
    layer for the output projection into the target vocab dimension.
    """
    def __init__(self, encoder, decoder):
        """
        Parameters
        ----------
        encoder
            An encoder class to use to encode input sequences.

        decoder
            A decoder class to use for decoding from encoder output.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_initial_decoder_output(self, context):
        """
        Parameters
        ----------
        context: Variable(FloatTensor)
            The encoder output to use in making the initial decoder output.
            Shape: (seq_len, batch_size, rnn_dim).

        Returns
        -------
        initial_decoder_output: Variable(FloatTensor)
            A zero-filled tensor of shape (batch_size, decoder_hidden_dim)
            that will be used as the first output of the decoder for input feeding.
        """
        batch_size = context.size(1)
        init_decoder_output_size = (batch_size, self.decoder.hidden_dim)
        return Variable(context.data.new(*init_decoder_output_size).zero_(),
                        requires_grad=False)

    def _fix_enc_hidden(self, encoder_hidden):
        """
        The hidden unit returned by the encoder is of shape
        (num_layers * num_directions, batch_size, hidden_dim),
        but we need to convert it to shape (num_layers, batch_size,
        hidden_dim * num_directions). Recall that the rnn_dim
        of the encoder is the value hidden_dim * num_directions

        Parameters
        ----------
        encoder_hidden: Variable(FloatTensor)
            The hidden unit of the encoder to reshape for compatibility
             with the decoder. Shape: (num_layers * num_directions, batch_size,
            hidden_dim)

        Returns
        -------
        reshaped_encoder_hidden: Variable(FloatTensor)
            A reshaped version of the encoder hidden unit that can be used with the
            decoder. Shape: (num_layers, batch_size, hidden_dim * num_directions)
        """
        if self.encoder.num_directions == 2:
            return torch.cat([encoder_hidden[0:encoder_hidden.size(0):2],
                              encoder_hidden[1:encoder_hidden.size(0):2]], 2)
        else:
            return encoder_hidden

    def forward(self, source_sequence, source_lengths, target_sequence):
        """
        Parameters
        ----------
        source_sequence: Variable(LongTensor)
            The source sequence to encode. Shape: (src_seq_len, batch_size)

        source_lengths: LongTensor
            The lengths of each timestep of the source sequence. Length batch_size.

        target_sequence: Variable(LongTensor)
            The target sequence. Shape: (tgt_seq_len, batch_size)

        Returns
        -------
        decoder_output: Variable(FloatTensor)
            A tensor of shape (seq_len, batch_size, rnn_dim) with the decoder output.
        """
        context, enc_hidden, context_lens = self.encoder(source_sequence, source_lengths)
        initial_output = self.make_initial_decoder_output(context)

        if isinstance(enc_hidden, tuple):
            enc_hidden = tuple(self._fix_enc_hidden(enc_hidden[i])
                               for i in range(len(enc_hidden)))
        else:
            enc_hidden = self._fix_enc_hidden(enc_hidden)

        # Exclude eos from target_sequence inputs
        target_sequence = target_sequence[:-1]
        out, dec_hidden, _attn = self.decoder(target_sequence, context, context_lens,
                                              enc_hidden, initial_output)

        return out
