import torch
import torch.nn as nn
from torch.autograd import Variable


def get_sequence_mask_from_lengths(sequence_lengths):
    """
    Given a list of lengths for a sequence, create a 2D
    mask for the sequence.

    Parameters
    ----------
    sequence_lengths: List of int
        List of int, where the length should be equal to the batch size.
        Each element at index i of sequence_lengths (sequence_lengths[i])
        corresponds to the length of the sequence (without padding) at
        batch[i].

    Returns
    -------
    mask: ByteTensor
        A ByteTensor of shape (batch_size, max_seq_len), where
        there is a 1 for valid elements of the sequence and a 0 for the
        padding of the sequence.
    """
    batch_size = len(sequence_lengths)
    max_sequence_length = max(sequence_lengths)

    # Create a zero tensor of the proper mask shape.
    mask = torch.ByteTensor(batch_size, max_sequence_length).zero_()
    if torch.cuda.is_available():
        mask = mask.cuda()
    else:
        mask = mask.cpu()

    # Iterate over the lengths and fill in the 1.0's
    for batch_idx, sequence_length in enumerate(sequence_lengths):
        mask[batch_idx, :sequence_length] = 1
    return mask


def pad_list(data_list, pad_value=0):
    """
    Given a list of 2D tensors, pad them to the an equal length.
    The dimension at index 0 is assumed to be the length.

    Parameters
    ----------
    data_list: List of Tensors
        A list of tensors of shape (length, batch_size) to pad.

    pad_value: int
        The int value to insert as padding.

    Returns
    -------
    padded_data_list: List of Tensors
        The same list of tensors, but each tensor has been padded to the same
        length with pad_value.
    """
    sequence_lengths = [x.size(0) for x in data_list]
    max_sequence_length = max(sequence_lengths)
    batch_sizes = [x.size(1) for x in data_list]
    if torch.cuda.is_available():
        data_pad = [torch.zeros(max_sequence_length,
                                batch_size).long().cuda() + pad_value for
                    batch_size in batch_sizes]
    else:
        data_pad = [torch.zeros(max_sequence_length, batch_size).long() + pad_value for
                    batch_size in batch_sizes]
    for (idx, data), sequence_length in zip(enumerate(data_list), sequence_lengths):
        data_pad[idx][:sequence_length] = data
    return data_pad


def NMTCriterion(vocab_size, padding_idx):
    """
    The criterion used for neural machine translation, which is
    summed negative log likelihood across all tokens in the batch,
    such that it is easy to use the loss to compute perplexity (math.exp(loss)).

    Parameters
    ----------
    vocab_size: int
        The target vocabulary size.

    padding_idx: int
        The index that corresponds to the padding index in the target
        language.
    """
    weight = torch.ones(vocab_size)
    weight[padding_idx] = 0
    criterion = nn.NLLLoss(weight, size_average=False)
    if torch.cuda.is_available():
        criterion.cuda()
    else:
        criterion.cpu()
    return criterion


def sort_mt_batch(source, target=None):
    """
    PyTorch's variable length RNNs require PackedSequences, which require that our batches
    be sorted by source sentence length. This function takes in the source / target
    batches (with data and lengths for each) and sorts the batch such that the
    source sentences are in decreasing order of length.

    Parameters
    ----------
    source: tuple
        A tuple of (data, lengths) for the source. The batch will be sorted such that
        the input lengths are in decreasing order.

    target: tuple, optional (default=None)
        A tuple of (data, lengths) for the target that will be moved around such that
        the source examples correspond to the same target examples in a batch.

    Returns
    -------
    sorted_batch:
        A tuple of ((src_data, src_lengths), (tgt_data, tgt_length)) after sorting if
        target is provided. Else, it is a tuple of (src_data, src_lengths).
    """
    input_data, input_lengths = source
    sorted_input_lengths, sorted_indices = torch.sort(
        input_lengths, -1, descending=True)

    if target is not None:
        target_data, target_lengths = target

    # Check if it was sorted to begin with
    if input_lengths.equal(sorted_input_lengths):
        if target is not None:
            return source, target
        else:
            return source

    sorted_input_data = Variable(input_data.data.gather(
        1, sorted_indices.expand_as(input_data)))
    if target is not None:
        sorted_target_lengths = target_lengths.gather(0, sorted_indices)
        sorted_target_data = Variable(target_data.data.gather(
            1, sorted_indices.expand_as(target_data)))
        return ((sorted_input_data, sorted_input_lengths),
                (sorted_target_data, sorted_target_lengths))
    else:
        return (sorted_input_data, sorted_input_lengths)


def sort_classification_batch(source, label=None):
    """
    PyTorch's variable length RNNs require PackedSequences, which require that our batches
    be sorted by source sentence length. This function takes in the source / target
    batches (with data and lengths for each) and sorts the batch such that the
    source sentences are in decreasing order of length.

    Parameters
    ----------
    source: tuple
        A tuple of (data, lengths) for the source. The batch will be sorted such that
        the input lengths are in decreasing order.

    label: LongTensor, optional (default=None)
        A Tensor holding the numeric labels for each instance.

    Returns
    -------
    sorted_batch:
        A tuple of ((src_data, src_lengths), (tgt_data, tgt_length)) after sorting if
        target is provided. Else, it is a tuple of (src_data, src_lengths).
    """
    input_data, input_lengths = source
    sorted_input_lengths, sorted_indices = torch.sort(
        input_lengths, -1, descending=True)

    # Check if it was sorted to begin with
    if input_lengths.equal(sorted_input_lengths):
        if label is not None:
            return source, label
        else:
            return source

    sorted_input_data = Variable(input_data.data.gather(
        1, sorted_indices.expand_as(input_data)))
    if label is not None:
        sorted_labels = Variable(label.data.gather(0, sorted_indices)) - 1
        return ((sorted_input_data, sorted_input_lengths),
                sorted_labels)
    else:
        return (sorted_input_data, sorted_input_lengths)


def get_rnn_class(type_str):
    if type_str == "LSTM":
        return nn.LSTM
    elif type_str == "GRU":
        return nn.GRU
    else:
        raise ValueError("Invalid RNN type string {}".format(type_str))
