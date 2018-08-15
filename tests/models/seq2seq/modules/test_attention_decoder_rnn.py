from ....common.test_case import OOVTestCase

from oov.models.seq2seq.modules.attention_decoder_rnn import AttentionDecoderRNN
from overrides import overrides
import torch
from torch.autograd import Variable


class TestAttentionDecoderRNN(OOVTestCase):
    @overrides
    def setUp(self):
        torch.manual_seed(42)
        super(TestAttentionDecoderRNN, self).setUp()
        self.batch_size = 3
        self.seq_len = 5

    def test_attention_decoder_basic(self):
        for rnn_type in ["LSTM", "GRU"]:
            embed_dim = 5
            rnn_dim = 6
            vocab_size = 10
            padding_idx = 0
            num_layers = 1
            attention_decoder_rnn = AttentionDecoderRNN(
                embed_dim=embed_dim,
                rnn_dim=rnn_dim,
                vocab_size=vocab_size,
                padding_idx=padding_idx,
                num_layers=num_layers,
                rnn_type=rnn_type)

            input_sequence = Variable(torch.LongTensor([[1, 1, 1],
                                                        [7, 4, 6],
                                                        [8, 5, 2],
                                                        [9, 2, 0],
                                                        [2, 0, 0]]))
            context = Variable(torch.rand(self.seq_len, self.batch_size,
                                          attention_decoder_rnn.rnn_dim))
            context_lengths = [5, 4, 2]
            if rnn_type == "LSTM":
                initial_hidden = [Variable(
                    torch.rand(num_layers, self.batch_size,
                               attention_decoder_rnn.rnn_dim)) for i in range(2)]
            else:
                initial_hidden = Variable(torch.rand(
                    num_layers, self.batch_size, attention_decoder_rnn.rnn_dim))
            initial_output = Variable(torch.rand(self.batch_size,
                                                 attention_decoder_rnn.hidden_dim))

            if torch.cuda.is_available():
                attention_decoder_rnn.cuda()
                input_sequence = input_sequence.cuda()
                context = context.cuda()
                if rnn_type == "LSTM":
                    for i in range(len(initial_hidden)):
                        initial_hidden[i] = initial_hidden[i].cuda()
                    initial_hidden = tuple(initial_hidden)
                elif rnn_type == "GRU":
                    initial_hidden = initial_hidden.cuda()

                initial_output = initial_output.cuda()

            outputs, hidden, attn_weights = attention_decoder_rnn(
                input_sequence, context, context_lengths, initial_hidden, initial_output)
            assert outputs.size() == (self.seq_len, self.batch_size,
                                      attention_decoder_rnn.rnn_dim)
            if rnn_type == "LSTM":
                assert len(hidden) == 2
                assert hidden[0].size() == (
                    num_layers, self.batch_size,
                    attention_decoder_rnn.rnn_dim)
                assert hidden[1].size() == (
                    num_layers, self.batch_size,
                    attention_decoder_rnn.rnn_dim)
            if rnn_type == "GRU":
                assert hidden.size() == (
                    num_layers, self.batch_size,
                    attention_decoder_rnn.rnn_dim)
            assert attn_weights.size() == (self.batch_size, self.seq_len)

    def test_attention_decoder_multilayer(self):
        for rnn_type in ["LSTM", "GRU"]:
            embed_dim = 5
            rnn_dim = 6
            vocab_size = 10
            padding_idx = 0
            num_layers = 3
            attention_decoder_rnn = AttentionDecoderRNN(
                embed_dim=embed_dim,
                rnn_dim=rnn_dim,
                vocab_size=vocab_size,
                padding_idx=padding_idx,
                num_layers=num_layers,
                rnn_type=rnn_type)

            input_sequence = Variable(torch.LongTensor([[1, 1, 1],
                                                        [7, 4, 6],
                                                        [8, 5, 2],
                                                        [9, 2, 0],
                                                        [2, 0, 0]]))
            context = Variable(torch.rand(self.seq_len, self.batch_size,
                                          attention_decoder_rnn.rnn_dim))
            context_lengths = [5, 4, 2]
            if rnn_type == "LSTM":
                initial_hidden = [Variable(
                    torch.rand(num_layers, self.batch_size,
                               attention_decoder_rnn.rnn_dim)) for i in range(2)]
            else:
                initial_hidden = Variable(torch.rand(
                    num_layers, self.batch_size, attention_decoder_rnn.rnn_dim))
            initial_output = Variable(torch.rand(self.batch_size,
                                                 attention_decoder_rnn.hidden_dim))

            if torch.cuda.is_available():
                attention_decoder_rnn.cuda()
                input_sequence = input_sequence.cuda()
                context = context.cuda()
                if rnn_type == "LSTM":
                    for i in range(len(initial_hidden)):
                        initial_hidden[i] = initial_hidden[i].cuda()
                    initial_hidden = tuple(initial_hidden)
                elif rnn_type == "GRU":
                    initial_hidden = initial_hidden.cuda()
                initial_output = initial_output.cuda()

            outputs, hidden, attn_weights = attention_decoder_rnn(
                input_sequence, context, context_lengths, initial_hidden, initial_output)
            assert outputs.size() == (self.seq_len, self.batch_size,
                                      attention_decoder_rnn.rnn_dim)
            if rnn_type == "LSTM":
                assert len(hidden) == 2
                assert hidden[0].size() == (
                    num_layers, self.batch_size,
                    attention_decoder_rnn.rnn_dim)
                assert hidden[1].size() == (
                    num_layers, self.batch_size,
                    attention_decoder_rnn.rnn_dim)
            if rnn_type == "GRU":
                assert hidden.size() == (
                    num_layers, self.batch_size,
                    attention_decoder_rnn.rnn_dim)
            assert attn_weights.size() == (self.batch_size, self.seq_len)
