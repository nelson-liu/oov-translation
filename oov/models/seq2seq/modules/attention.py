import torch
import torch.nn as nn


class Attention(nn.Module):
    """Implements Luong global attention with the general scoring function. For
    details, see section 3.1 of "Effective Approaches to Attention-based
    Neural Machine Translation" (Luong et al 2015,
    https://arxiv.org/pdf/1508.04025.pdf)
    """
    def __init__(self, rnn_dim):
        """
        Parameters
        ----------
        rnn_dim: int
            The dimensionality of the RNN outputs.
        """
        super(Attention, self).__init__()
        self.linear_in = nn.Linear(rnn_dim, rnn_dim, bias=False)
        # Instantiate layers instead of using functional API for softmax and
        # tanh so they show up in the Attention string representation.
        self.softmax = nn.Softmax()
        self.linear_out = nn.Linear(rnn_dim * 2, rnn_dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def set_mask(self, mask):
        """
        Parameters
        ----------
        mask: ByteTensor
            The mask to apply to their context vectors, such we do not give
            any attention weight to masked timesteps. This tensor should have
            1's in valid timesteps and 0's in padding timesteps.
            Shape: (batch_size, context_len)
        """
        self.mask = mask

    def forward(self, query, context):
        """
        Parameters
        ----------
        query: Variable(FloatTensor)
            The vector input to calculate the attention from.
            Shape: (batch_size, rnn_dim)

        context: Variable(FloatTensor)
            The matrix context to attend to and calculate a weighted
            representation of given the query vector.
            Shape: (batch_size, context_len, rnn_dim)

        Returns
        -------
        context_output: Variable(FloatTensor)
            The weighted context. Shape: (batch_size, rnn_dim)

        attn_weights: Variable(FloatTensor)
            The weights assigned to each element of the context.
            Shape: (batch_size, context_len)
        """
        # Shape: (batch_size, rnn_dim, 1)
        targetT = self.linear_in(query).unsqueeze(2)

        # Get attention
        # Shape: (batch_size, context_len)
        unnorm_attn = torch.bmm(context, targetT).squeeze(2)

        # Set the masked out values to -inf. We flip the mask first since
        # masked_fill fills the indices with 1's.
        if self.mask is not None:
            unnorm_attn.data.masked_fill_(1 - self.mask, -float("inf"))
        # Shape: (batch_size, context_len)
        attn_weights = self.softmax(unnorm_attn)
        # Shape: (batch_size, 1, context_len)
        expanded_attn = attn_weights.view(attn_weights.size(0), 1,
                                          attn_weights.size(1))

        # Shape: (batch_size, rnn_dim)
        weighted_context = torch.bmm(expanded_attn, context).squeeze(1)
        # Shape: (batch_size, rnn_dim*2)
        context_combined = torch.cat((weighted_context, query), 1)
        # Shape: (batch_size, rnn_dim)
        context_output = self.tanh(self.linear_out(context_combined))

        return context_output, attn_weights
