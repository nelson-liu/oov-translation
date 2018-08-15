from ....common.test_case import OOVTestCase

import numpy as np
from numpy.testing import assert_allclose
from oov.models.seq2seq.modules.encoder_rnn import EncoderRNN
from overrides import overrides
import torch
from torch.autograd import Variable


class TestEncoderRNN(OOVTestCase):
    @overrides
    def setUp(self):
        torch.manual_seed(42)
        super(TestEncoderRNN, self).setUp()
        self.embed_dim = 5
        self.rnn_dim = 10
        self.vocab_size = 20
        self.padding_idx = 0

    def test_encoder_RNN_single_layer(self):
        num_layers = 1
        encoder_rnn = EncoderRNN(
            embed_dim=self.embed_dim,
            rnn_dim=self.rnn_dim,
            vocab_size=self.vocab_size,
            padding_idx=self.padding_idx,
            num_layers=num_layers)
        batch_size = 3
        seq_len = 4
        input_words = Variable(torch.LongTensor([[3, 1, 7],
                                                 [2, 6, 0],
                                                 [1, 2, 0],
                                                 [1, 0, 0]]))
        input_lengths = torch.LongTensor([4, 3, 1])

        if torch.cuda.is_available():
            encoder_rnn.cuda()
            input_words = input_words.cuda()

        encoder_output, encoder_hidden, context_len = encoder_rnn(input_words,
                                                                  input_lengths)

        # Check that the output of the encoder has proper shape
        assert encoder_output.size() == (seq_len, batch_size, encoder_rnn.rnn_dim)

        # Check that the output of the encoder is properly masked w.r.t to the input
        assert_allclose(encoder_output[1][2].data.cpu().numpy(),
                        np.zeros(self.rnn_dim))
        assert_allclose(encoder_output[2][2].data.cpu().numpy(),
                        np.zeros(self.rnn_dim))
        assert_allclose(encoder_output[3][1].data.cpu().numpy(),
                        np.zeros(self.rnn_dim))
        assert_allclose(encoder_output[3][2].data.cpu().numpy(),
                        np.zeros(self.rnn_dim))

        # Check that the encoder output hidden layer has proper shape
        assert len(encoder_hidden) == 2
        assert encoder_hidden[0].size() == (
            encoder_rnn.num_layers * encoder_rnn.num_directions,
            batch_size,
            encoder_rnn.hidden_dim)
        assert encoder_hidden[1].size() == (
            encoder_rnn.num_layers * encoder_rnn.num_directions,
            batch_size,
            encoder_rnn.hidden_dim)

        assert context_len == [4, 3, 1]

    def test_encoder_RNN_multilayer(self):
        for rnn_type in ["LSTM", "GRU"]:
            encoder_rnn = EncoderRNN(
                embed_dim=self.embed_dim,
                rnn_dim=self.rnn_dim,
                vocab_size=self.vocab_size,
                padding_idx=self.padding_idx,
                num_layers=3,
                rnn_type=rnn_type)
            batch_size = 3
            seq_len = 4
            input_words = Variable(torch.LongTensor([[3, 1, 7],
                                                     [2, 6, 0],
                                                     [1, 2, 0],
                                                     [1, 0, 0]]))
            input_lengths = torch.LongTensor([4, 3, 1])

            if torch.cuda.is_available():
                encoder_rnn.cuda()
                input_words = input_words.cuda()

            encoder_output, encoder_hidden, context_len = encoder_rnn(input_words,
                                                                      input_lengths)

            # Check that the output of the encoder has proper shape
            assert encoder_output.size() == (seq_len, batch_size, encoder_rnn.rnn_dim)

            # Check that the output of the encoder is properly masked w.r.t to the input
            assert_allclose(encoder_output[1][2].data.cpu().numpy(),
                            np.zeros(self.rnn_dim))
            assert_allclose(encoder_output[2][2].data.cpu().numpy(),
                            np.zeros(self.rnn_dim))
            assert_allclose(encoder_output[3][1].data.cpu().numpy(),
                            np.zeros(self.rnn_dim))
            assert_allclose(encoder_output[3][2].data.cpu().numpy(),
                            np.zeros(self.rnn_dim))

            # Check that the encoder output hidden layer has proper shape
            if rnn_type == "LSTM":
                assert len(encoder_hidden) == 2
                assert encoder_hidden[0].size() == (
                    encoder_rnn.num_layers * encoder_rnn.num_directions,
                    batch_size,
                    encoder_rnn.hidden_dim)
                assert encoder_hidden[1].size() == (
                    encoder_rnn.num_layers * encoder_rnn.num_directions,
                    batch_size,
                    encoder_rnn.hidden_dim)
            if rnn_type == "GRU":
                assert encoder_hidden.size() == (
                    encoder_rnn.num_layers * encoder_rnn.num_directions,
                    batch_size,
                    encoder_rnn.hidden_dim)

            assert context_len == [4, 3, 1]
