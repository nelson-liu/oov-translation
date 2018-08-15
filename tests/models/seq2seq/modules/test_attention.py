from ....common.test_case import OOVTestCase

from oov.models.seq2seq.modules.attention import Attention
from overrides import overrides
import torch
from torch.autograd import Variable


class TestAttention(OOVTestCase):
    @overrides
    def setUp(self):
        torch.manual_seed(42)
        super(TestAttention, self).setUp()
        self.batch_size = 3
        self.context_len = 4

    def test_attention(self):
        rnn_dim = 5
        attention = Attention(rnn_dim=rnn_dim)

        query_vector = Variable(torch.rand(self.batch_size, rnn_dim))
        context_matrix = Variable(torch.rand(self.batch_size, self.context_len, rnn_dim))
        mask = torch.ByteTensor([[1, 1, 1, 0],
                                 [1, 1, 0, 0],
                                 [1, 0, 0, 0]])

        if torch.cuda.is_available():
            attention.cuda()
            query_vector = query_vector.cuda()
            context_matrix = context_matrix.cuda()
            mask = mask.cuda()

        attention.set_mask(mask)

        context_output, attn_weights = attention(query_vector, context_matrix)
        # Check that the context_output has the proper shape
        assert context_output.size() == (self.batch_size, rnn_dim)

        # Check that the attn_weights have the proper shape
        assert attn_weights.size() == (self.batch_size, self.context_len)

        # Test that the mask was properly applied
        assert attn_weights.data[0][3] == 0
        assert attn_weights.data[1][2] == 0
        assert attn_weights.data[1][3] == 0
        assert attn_weights.data[2][1] == 0
        assert attn_weights.data[2][2] == 0
        assert attn_weights.data[2][3] == 0
