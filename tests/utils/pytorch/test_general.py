from ...common.test_case import OOVTestCase

import torch
from torch.autograd import Variable
import numpy as np
from numpy.testing import assert_allclose
from oov.utils.pytorch.general import (
    get_sequence_mask_from_lengths, pad_list, sort_mt_batch)


class TestPyTorchGeneralUtils(OOVTestCase):
    def test_get_sequence_mask_from_lengths(self):
        lengths = [5, 2, 4, 3, 1]
        mask = get_sequence_mask_from_lengths(lengths)
        assert_allclose(mask[0].cpu().numpy(), np.ones(5))
        assert_allclose(mask[1].cpu().numpy(), np.array([1, 1, 0, 0, 0]))
        assert_allclose(mask[2].cpu().numpy(), np.array([1, 1, 1, 1, 0]))
        assert_allclose(mask[3].cpu().numpy(), np.array([1, 1, 1, 0, 0]))
        assert_allclose(mask[4].cpu().numpy(), np.array([1, 0, 0, 0, 0]))

    def test_pad_list(self):
        # Shapes should be (len, batch_size)
        # 6 x 2
        lang_one = torch.LongTensor([[1, 8], [2, 2], [3, 37], [1, 1], [5, 9],
                                     [2, 100]])
        # 3 x 4
        lang_two = torch.LongTensor([[3, 2, 1, 1],
                                     [1, 6, 2, 100],
                                     [7, 100, 100, 100]])
        # 2 x 1
        lang_three = torch.LongTensor([[1, 2]])

        if torch.cuda.is_available():
            lang_one = lang_one.cuda()
            lang_two = lang_two.cuda()
            lang_three = lang_three.cuda()
        padded = pad_list([lang_one, lang_two, lang_three], 100)
        assert len(padded) == 3

        expected_lang_one = torch.LongTensor(
            [[1, 8], [2, 2], [3, 37],
             [1, 1], [5, 9], [2, 100]])
        expected_lang_two = torch.LongTensor([[3, 2, 1, 1],
                                              [1, 6, 2, 100],
                                              [7, 100, 100, 100],
                                              [100, 100, 100, 100],
                                              [100, 100, 100, 100],
                                              [100, 100, 100, 100]])
        expected_lang_three = torch.LongTensor([[1, 2],
                                                [100, 100],
                                                [100, 100],
                                                [100, 100],
                                                [100, 100],
                                                [100, 100]])
        if torch.cuda.is_available():
            expected_lang_one = expected_lang_one.cuda()
            expected_lang_two = expected_lang_two.cuda()
            expected_lang_three = expected_lang_three.cuda()

        assert torch.equal(padded[0], expected_lang_one)
        assert torch.equal(padded[1], expected_lang_two)
        assert torch.equal(padded[2], expected_lang_three)

    def test_sort_mt_batch(self):
        input_words = Variable(torch.LongTensor([[3, 1, 7],
                                                 [2, 6, 0],
                                                 [1, 2, 0],
                                                 [1, 0, 0]]))
        input_lengths = torch.LongTensor([4, 3, 1])
        if torch.cuda.is_available():
            input_words = input_words.cuda()
            input_lengths = input_lengths.cuda()
        sorted_batch = sort_mt_batch((input_words, input_lengths))
        sorted_input_words, sorted_input_lengths = sorted_batch
        assert torch.equal(sorted_input_words.data, input_words.data)
        assert torch.equal(sorted_input_lengths, input_lengths)

        input_words = Variable(torch.LongTensor([[1, 3, 7],
                                                 [6, 2, 0],
                                                 [2, 1, 0],
                                                 [0, 1, 0]]))
        input_lengths = torch.LongTensor([3, 4, 1])
        expected_sorted_input_words = Variable(
            torch.LongTensor([[3, 1, 7],
                              [2, 6, 0],
                              [1, 2, 0],
                              [1, 0, 0]]))
        expected_sorted_input_lengths = torch.LongTensor([4, 3, 1])

        if torch.cuda.is_available():
            input_words = input_words.cuda()
            input_lengths = input_lengths.cuda()
            expected_sorted_input_words = expected_sorted_input_words.cuda()
            expected_sorted_input_lengths = expected_sorted_input_lengths.cuda()
        sorted_batch = sort_mt_batch((input_words, input_lengths))
        sorted_input_words, sorted_input_lengths = sorted_batch
        assert torch.equal(sorted_input_words.data,
                           expected_sorted_input_words.data)
        assert torch.equal(sorted_input_lengths,
                           expected_sorted_input_lengths)

        input_words = Variable(torch.LongTensor([[1, 3, 7],
                                                 [6, 2, 0],
                                                 [2, 1, 0],
                                                 [0, 1, 0]]))
        input_lengths = torch.LongTensor([3, 4, 1])
        target_words = Variable(torch.LongTensor([[1, 1, 7],
                                                  [0, 2, 3],
                                                  [0, 0, 6]]))
        target_lengths = torch.LongTensor([1, 2, 3])
        expected_sorted_input_words = Variable(
            torch.LongTensor([[3, 1, 7],
                              [2, 6, 0],
                              [1, 2, 0],
                              [1, 0, 0]]))
        expected_sorted_input_lengths = torch.LongTensor([4, 3, 1])
        expected_sorted_target_words = Variable(
            torch.LongTensor([[1, 1, 7],
                              [2, 0, 3],
                              [0, 0, 6]]))
        expected_sorted_target_lengths = torch.LongTensor([2, 1, 3])
        if torch.cuda.is_available():
            input_words = input_words.cuda()
            input_lengths = input_lengths.cuda()
            target_words = target_words.cuda()
            target_lengths = target_lengths.cuda()
            expected_sorted_input_words = expected_sorted_input_words.cuda()
            expected_sorted_input_lengths = expected_sorted_input_lengths.cuda()
            expected_sorted_target_words = expected_sorted_target_words.cuda()
            expected_sorted_target_lengths = expected_sorted_target_lengths.cuda()

        sorted_inputs, sorted_targets = sort_mt_batch(
            (input_words, input_lengths), (target_words, target_lengths))
        sorted_input_words, sorted_input_lengths = sorted_inputs
        assert torch.equal(sorted_input_words.data,
                           expected_sorted_input_words.data)
        assert torch.equal(sorted_input_lengths,
                           expected_sorted_input_lengths)
        sorted_target_words, sorted_target_lengths = sorted_targets
        assert torch.equal(sorted_target_words.data,
                           expected_sorted_target_words.data)
        assert torch.equal(sorted_target_lengths,
                           expected_sorted_target_lengths)
