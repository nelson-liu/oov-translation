from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn

import logging
import math
import operator
import os
import six
import sys
from random import randint

import dill
from overrides import overrides
from torchtext import data
from tqdm import trange, tqdm


from .modules.encoder_rnn import EncoderRNN
from .modules.attention_decoder_rnn import AttentionDecoderRNN
from .modules.seq2seq import Seq2Seq
from ..base_solver import BaseSolver
from ...utils.pytorch.general import NMTCriterion, pad_list, sort_mt_batch
from ...utils.pytorch.optimizer import Optimizer
from ...utils.pytorch.tensorboard_logger import TensorboardLogger

logger = logging.getLogger(__name__)


class BaseSeq2SeqTranslationModel(BaseSolver):
    """
    Base class for Seq2Seq translation models. Do not use this class
    directly, instead use one of its subclasses.
    """
    def __init__(self, batch_size, embed_dim, rnn_dim,
                 output_projection_batch_size, dropout=0.3,
                 num_layers=2, bidirectional=True, input_feed=True,
                 rnn_type="LSTM", max_gradient_norm=5, param_init=None,
                 optimizer="adam", lr=0.001, lr_decay=None,
                 start_decay_at=None, seed=0):
        """Create a seq2seq translation model.

        Parameters
        ----------
        batch_size: int
            The number of examples per batch.

        embed_dim: int
            The dimensionality of the embedding layer.

        rnn_dim: int
            The dimensionality of the decoder and encoder RNN outputs.

        output_projection_batch_size: int
            The number of batches to process at once with the output_projection.
            Lower is more memory-conserving, but also slower.

        dropout: float, optional (default=0.3)
            The proportion of RNN outputs to drop out after each layer
            except the last.

        num_layers: int, optinal (default=2)
            The number of layers to stack in the encoder and decoder RNN.

        bidirectional: boolean, optional (default=True)
            Whether not to use a bidirectional encoder.

        input_feed: boolean, optional (default=True)
            Whether or not to concatenate the context vector to the input
            at the next timestep.

        rnn_type: str, optional (default="LSTM")
            One of [LSTM|GRU] to indicate the type of RNN to use.

        max_gradient_norm: float, optional (default=5.0)
            If the gradient vector norm exceeds this value,
            renormalize the gradients to have norm equal to max_grad_norm.

        param_init: float, optional (default=None)
            If not none, all model parameters are initialized with the random uniform
            distribution (-param_init, param_init)

        optimizer: string, optional (default=Adam)
            The optimizer to use. Options are [sgd|adagrad|adadelta|adam|yellowfin].

        lr: float, optional (default=0.001)
            The learning rate to use for the optimizer.

        lr_decay: float, optional (default=None)
            Decay learning rate by this much if perplexity doesn't increase on
            the validation set or if epoch has gone past start_decay_at.
            If None, does not decay.

        start_decay_at: int, optional (default=None)
            Start learning rate decay every epoch after and including this one.
            If None, does not decay

        seed: int, optional (default=0)
            The PyTorch random seed. If None, the seed is randomly chosen.
        """
        self.solver_init_params = locals()
        self.solver_init_params.pop("self")
        self.solver_init_params.pop("__class__", None)

        self.seed = seed if seed is not None else randint(0, 10e5)
        if seed is not None:
            logger.info("Using provided seed {}".format(self.seed))
        else:
            logger.info("No seed provided, using randomly "
                        "generated seed {}".format(self.seed))
        torch.manual_seed(self.seed)
        super(BaseSeq2SeqTranslationModel, self).__init__()
        self.batch_size = batch_size
        self.output_projection_batch_size = output_projection_batch_size
        self.max_gradient_norm = max_gradient_norm
        self.param_init = param_init
        self.rnn_type = rnn_type
        self.optimizer_str = optimizer
        self.lr = lr
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at

        # TorchText fields that hold the vocabulary / map them to indices
        self.source_field = None
        self.target_field = None

        # Encoder/Decoder shared parameters
        self.embed_dim = embed_dim
        self.rnn_dim = rnn_dim
        self.dropout = dropout
        self.num_layers = num_layers

        # Encoder-specific parameters
        self.bidirectional = bidirectional
        self.src_vocab_size = None
        self.src_padding_idx = None

        # Decoder-specific parameters
        self.input_feed = input_feed
        self.target_vocab_size = None
        self.target_init_idx = None
        self.target_padding_idx = None
        self.target_eos_idx = None

        # Temporary placeholder variables for the encoder, decoder,
        # optimizers, and criterion. We will instantiate these objects
        # in _build_model.
        self.model = None
        self.criterion = None
        self.optimizer = None

        # Keep track of the number of train steps completed
        self.global_step = 0
        # Keep track of how many epochs completed
        self.epoch_counter = 0

    @overrides
    def get_state_dict(self):
        model_state_dict = (self.model.state_dict())
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'output_projection' not in k}
        output_proj_state_dict = self.model.output_projection.state_dict()
        state_dict = {
            "solver_class": self.__class__,
            "solver_init_params": self.solver_init_params,
            "model": model_state_dict,
            "optimizer": self.optimizer,
            "output_projection": output_proj_state_dict,
            "source_field": self.source_field,
            "target_field": self.target_field,
            "global_step": self.global_step,
            "epoch_counter": self.epoch_counter,
            "seed": self.seed
        }
        return state_dict

    @overrides
    def load_from_state_dict(self, state_dict):
        self.seed = state_dict["seed"]
        torch.manual_seed(self.seed)
        logger.info("Manually seeding with seed "
                    "from state dict {}".format(self.seed))

        # Populate data-related members
        self.source_field = state_dict["source_field"]
        self.target_field = state_dict["target_field"]
        self.src_vocab_size = len(self.source_field.vocab)
        self.src_padding_idx = self.source_field.vocab.stoi['<pad>']
        self.target_vocab_size = len(self.target_field.vocab)
        self.target_init_idx = self.target_field.vocab.stoi['<bos>']
        self.target_padding_idx = self.target_field.vocab.stoi['<pad>']
        self.target_eos_idx = self.target_field.vocab.stoi['<eos>']

        # Instantiate the model
        encoder = EncoderRNN(
            embed_dim=self.embed_dim,
            rnn_dim=self.rnn_dim,
            vocab_size=self.src_vocab_size,
            padding_idx=self.src_padding_idx,
            dropout=self.dropout,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            rnn_type=self.rnn_type)
        decoder = AttentionDecoderRNN(
            embed_dim=self.embed_dim,
            rnn_dim=self.rnn_dim,
            vocab_size=self.target_vocab_size,
            padding_idx=self.target_padding_idx,
            dropout=self.dropout,
            num_layers=self.num_layers,
            input_feed=self.input_feed,
            rnn_type=self.rnn_type)
        output_projection = nn.Sequential(
            nn.Linear(self.rnn_dim, self.target_vocab_size),
            nn.LogSoftmax())
        self.model = Seq2Seq(encoder, decoder)
        # Load the model parameters
        self.model.load_state_dict(state_dict["model"])
        output_projection.load_state_dict(state_dict["output_projection"])

        # Move model to GPU if is CUDA is available.
        if torch.cuda.is_available():
            self.model.cuda()
            output_projection.cuda()
        else:
            self.model.cpu()
            output_projection.cpu()

        # Set the output projection of the model. This isn't an init argument because
        # there are some parallelism differences between it and the encoder/decoder
        self.model.output_projection = output_projection

        # Create the optimizer and load its state dict.
        self.optimizer = Optimizer(
            method=self.optimizer_str, lr=self.lr,
            max_gradient_norm=self.max_gradient_norm,
            lr_decay=self.lr_decay, start_decay_at=self.start_decay_at)

        self.optimizer.set_parameters(self.model.parameters())
        # Load optimizer state dict if applicable
        if hasattr(self.optimizer.optimizer, "load_state_dict"):
            self.optimizer.optimizer.load_state_dict(
                state_dict["optimizer"].optimizer.state_dict())

        # Set the global step
        self.global_step = state_dict["global_step"]
        # Set the epoch counter
        self.epoch_counter = state_dict["epoch_counter"]

        num_params = sum([p.nelement() for p in self.model.parameters()])
        logger.info("Loaded Model: \n {}".format(self.model))
        logger.info("Number of parameters: {}".format(num_params))
        logger.info("Steps Completed: {}".format(self.global_step))
        logger.info("Epochs Completed: {}".format(self.epoch_counter))
        return self

    @overrides
    def load_from_file(self, checkpoint_path):
        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage)
        return self.load_from_state_dict(state_dict)

    @overrides
    def read_data(self, train_file, val_file, max_source_vocab=50000,
                  max_target_vocab=50000):
        """
        Given the path to a TSV train and val files where each line is of
        format <source><tab><target>, read the data and return a tuple of
        iterator.

        Parameters
        ----------
        train_file: str
            The string path to the file with the train data.

        val_file: str
            The string path to the file with the validation data.

        max_source_vocab: int, optional (default=50000)
            The maximum size of the source vocabulary.

        max_target_vocab: int, optional (default=50000)
            The maximum size of the target vocabulary.

        Returns
        -------
        data
            A data object that can be consumed directly by the train_model()
            function, for example a tuple of (train, validation)
            iterators.
        """
        # Subclasses should override this method, construct Fields
        # to fill self.source_field and self.target_field with their
        # appropriate tokenizers, and then call the super class.
        train = data.TabularDataset(
            path=train_file, format='tsv',
            fields=[("source", self.source_field),
                    ("target", self.target_field)])
        val = data.TabularDataset(
            path=val_file, format='tsv',
            fields=[("source", self.source_field),
                    ("target", self.target_field)])

        if not hasattr(self.source_field, "vocab"):
            self.source_field.build_vocab(train, max_size=max_source_vocab)
        if not hasattr(self.target_field, "vocab"):
            self.target_field.build_vocab(train, max_size=max_target_vocab)

        # Create batches on current GPU if CUDA is available, else CPU
        device = None if torch.cuda.is_available() else -1

        train_iter = data.BucketIterator(
            dataset=train, batch_size=self.batch_size, device=device,
            sort_key=lambda x: data.interleave_keys(len(x.source),
                                                    len(x.target)))
        val_iter = data.BucketIterator(
            dataset=val, batch_size=self.batch_size, train=False, device=device,
            sort_key=lambda x: data.interleave_keys(len(x.source),
                                                    len(x.target)))

        if self.src_vocab_size is None:
            self.src_vocab_size = len(self.source_field.vocab)
        if self.src_padding_idx is None:
            self.src_padding_idx = self.source_field.vocab.stoi['<pad>']
        if self.target_vocab_size is None:
            self.target_vocab_size = len(self.target_field.vocab)
        if self.target_init_idx is None:
            self.target_init_idx = self.target_field.vocab.stoi['<bos>']
        if self.target_padding_idx is None:
            self.target_padding_idx = self.target_field.vocab.stoi['<pad>']
        if self.target_eos_idx is None:
            self.target_eos_idx = self.target_field.vocab.stoi['<eos>']
        return {"train_iter": train_iter, "val_iter": val_iter}

    @overrides
    def save_to_file(self, save_dir, run_id, checkpoint_name=None):
        """
        Parameters
        ----------
        save_dir: str
            The folder to save the trained model in.

        run_id: str
            The run_id for this model, also dictates the serialization filename.

        checkpoint_name: str, default=None
            If specified, use this name as the checkpoint name instead of generating
            one from the run_id and current global step count.

        Returns
        -------
        checkpoint_path: str
            The path that the checkpoint was saved to.
        """
        if checkpoint_name is None:
            checkpoint_name = run_id + "-step-" + str(self.global_step) + ".ckpt"
        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        state_dict = self.get_state_dict()
        torch.save(state_dict, checkpoint_path, pickle_module=dill)
        return checkpoint_path

    @overrides
    def train_model(self, train_iter, val_iter, num_epochs=15, log_period=50,
                    val_period=500, log_dir=None, save_dir=None, run_id=None,
                    early_stopping_patience=5, show_progbar=True, max_to_keep=5):
        """
        Train the model, given data and some training hyperparameters.

        Parameters
        ----------
        train_iter: Iterable or Dict of Iterable
            An iterable with the training data, or a dict of such iterables.
            If a dictionary is provided, we assume it is a dictionary of
            language to iterator, and will combine the batches from each of the
            iterators before feeding them into the model.

        val_iter: Iterable or Dict of Iterable
            An iterable with the validation data, or a dict of such iterables.
            If a dictionary is provided, we evaluate validation loss separately
            on each of the provided iterables.

        num_epochs: int, optional (default=15)
            The number of training epochs to run.

        log_period: int, optional (default=50)
            The number of train steps between writing progress to a log.
            If log_dir is None, no logs are written.

        val_period: int, optional (default=500)
            The number of train steps between each evaluation on the
            validation set. If log_dir is None, no validation is run.

        log_dir: str, optional (default=None)
            The folder to which to save logs. We create separate train and val
            folders in this directory to hold train and validation logs.

        save_dir: str, optional (default=None)
            The folder to save model checkpoints to. If this is provided, you
            must also provide a value for run_id.

        run_id: str, optional (default=None)
            The run_id for this experiment. This is used to identify the
            checkpoint files.

        early_stopping_patience: int, optional (default=5)
            An integer (0 or higher) that describes how many consecutive
            epochs of no reduction in validation loss must be observed
            in order to stop early.

        show_progbar: boolean, optional (default=True)
            Whether or not to show a progress bar during training.

        max_to_keep: int, optional (default=5)
            The number of checkpoints to keep. Checkpoints will be kept based on
            their sequence validation accuracy (so the best n performing models on
            validation). If None or 0, keep all.
        """
        if max_to_keep is None or max_to_keep == 0:
            max_to_keep = six.MAXSIZE

        # Make sure that we called read_data before training
        if self.source_field is None or self.target_field is None:
            raise ValueError(
                "Couldn't find source_field and target_field "
                "objects --- note that you must call read_data() before "
                "calling train().")

        # If save_dir is provided, run_id must be provided as well
        if save_dir is not None and run_id is None:
            raise ValueError("Must pass in run_id if you "
                             "wish to save checkpoints.")

        self._build_model()
        logger.info("Model: \n {}".format(self.model))
        self.model.train()

        # If we were provided a log_dir, create a TensorboardLogger for
        # train and val
        if log_dir is not None:
            train_tensorboard_logger = TensorboardLogger.from_dir(
                log_dir, "train", start_step=self.global_step)
            val_tensorboard_logger = TensorboardLogger.from_dir(
                log_dir, "val", start_step=self.global_step)

        # Store a mapping from checkpoint filepath to its accuracy
        saved_checkpoints = {}

        # Iterate over the epochs
        epoch_val_accs = []
        if show_progbar:
            epoch_iterator = trange(
                self.epoch_counter, self.epoch_counter + num_epochs,
                desc="Epoch", file=sys.stdout)
            epoch_iterator.set_description("Completed {} Epochs".format(
                self.epoch_counter))
        else:
            epoch_iterator = range(self.epoch_counter, self.epoch_counter + num_epochs)
        for epoch_num in epoch_iterator:
            # Iterate over the training batches
            if log_dir is not None:
                log_period_total_loss, log_period_num_correct = 0, 0
                log_period_num_target_words = 0
            if show_progbar:
                if isinstance(train_iter, dict):
                    # Get length of zip (length of shortest iterator)
                    # without making a list
                    if six.PY2:
                        raise ValueError("{} can only be run with Python 3".format(
                            self.__class__))
                    train_len = min([len(data_iter) for data_iter in train_iter.values()])
                    train_batches_iterator = tqdm(
                        enumerate(zip(*train_iter.values())), total=train_len,
                        file=sys.stdout, initial=int(self.global_step % train_len),
                        desc="Train Batches")
                else:
                    train_len = len(train_iter)
                    train_batches_iterator = tqdm(
                        enumerate(train_iter), total=train_len, file=sys.stdout,
                        initial=int(self.global_step % train_len), desc="Train Batches")
            else:
                if isinstance(train_iter, dict):
                    if six.PY2:
                        raise ValueError("{} can only be run with Python 3".format(
                            self.__class__))
                    train_len = min([len(data_iter) for data_iter in train_iter.values()])
                    train_batches_iterator = enumerate(zip(*train_iter.values()))
                else:
                    train_len = len(train_iter)
                    train_batches_iterator = enumerate(train_iter)

            train_scalar_dicts = []
            for train_batch_idx, train_batch in train_batches_iterator:
                if isinstance(train_batch, tuple) and isinstance(train_iter, dict):
                    # Concatenate the batches generated by the iterators for each lang
                    # Concatenate the source lengths, padding with self.src_padding_idx
                    # when necessary
                    combined_source_lens = torch.cat([x.source[1] for x in train_batch])
                    # Concatenate the source data tensors
                    padded_source = Variable(
                        torch.cat(pad_list([x.source[0].data for x in train_batch],
                                           pad_value=self.src_padding_idx), 1))
                    # Concatenate the target lengths, padding with self.src_padding_idx
                    # when necessary
                    combined_target_lens = torch.cat([x.target[1] for x in train_batch])
                    # Concatenate the target data tensors
                    padded_target = Variable(
                        torch.cat(pad_list([x.target[0].data for x in train_batch],
                                           pad_value=self.src_padding_idx), 1))
                    # Wrap it all together and sort
                    train_batch_source, train_batch_target = sort_mt_batch(
                        (padded_source, combined_source_lens),
                        (padded_target, combined_target_lens))
                else:
                    train_batch_source, train_batch_target = sort_mt_batch(
                        train_batch.source, train_batch.target)

                source_data, source_lengths = train_batch_source
                target_data, _ = train_batch_target

                self.model.zero_grad()
                outputs = self.model(source_data, source_lengths, target_data)
                # Exclude target init token from targets.
                target_data = target_data[1:]
                loss, gradient, num_correct = self._calculate_loss(
                    outputs, target_data, self.model.output_projection,
                    self.criterion)
                # Backwards pass
                outputs.backward(gradient)

                # Update the parameters
                self.optimizer.step()
                self.global_step += 1

                # Accumulate total statistics and statistics for this log period
                if log_dir is not None:
                    num_target_words = target_data.data.ne(self.target_padding_idx).sum()
                    log_period_total_loss += loss
                    log_period_num_correct += num_correct
                    log_period_num_target_words += num_target_words

                    # Write the train metrics to Tensorboard
                    if self.global_step % log_period == 0:
                        log_period_token_acc = (log_period_num_correct /
                                                log_period_num_target_words)
                        log_period_avg_token_loss = (log_period_total_loss /
                                                     log_period_num_target_words)
                        log_period_ppl = math.exp(min(log_period_avg_token_loss, 100))

                        train_scalar_dict = {
                            "train log period loss": log_period_avg_token_loss,
                            "train log period perplexity": log_period_ppl,
                            "train token-level accuracy": log_period_token_acc}
                        train_scalar_dicts.append((train_scalar_dict, self.global_step))
                        log_period_total_loss, log_period_num_correct = 0, 0
                        log_period_num_target_words = 0

                # Evaluate on val set.
                if (self.global_step % val_period == 0 or
                        self.global_step % train_len == 0):
                    if self.global_step % train_len == 0:
                        self.epoch_counter += 1

                    def evaluate_and_log(lang_val_iter, lang=None):
                        val_loss, val_token_acc, val_seq_acc = self._evaluate(
                            lang_val_iter, show_progbar=show_progbar)
                        val_ppl = math.exp(min(val_loss, 100))
                        # Write the val metrics to Tensorboard
                        if log_dir is not None:
                            prefix = lang + " val" if lang else "val"
                            val_scalar_dict = {
                                "{} loss".format(prefix): val_loss,
                                "{} perplexity".format(prefix): val_ppl,
                                "{} token-level accuracy".format(prefix):
                                val_token_acc,
                                "{} sequence-level accuracy".format(prefix):
                                val_seq_acc}
                            val_tensorboard_logger.scalar_summary(
                                val_scalar_dict, self.global_step)
                        return val_ppl, val_seq_acc

                    if isinstance(val_iter, dict):
                        # val_seq_acc is the average seq acc across languages
                        val_seq_acc = 0
                        val_ppl = 0
                        for lang, lang_val_iter in val_iter.items():
                            lang_val_ppl, lang_val_seq_acc = evaluate_and_log(
                                lang_val_iter, lang)
                            val_seq_acc += lang_val_seq_acc
                            val_ppl += lang_val_ppl
                        val_seq_acc /= len(val_iter)
                        val_ppl /= len(val_iter)
                    else:
                        val_ppl, val_seq_acc = evaluate_and_log(val_iter)

                    if log_dir is not None:
                        # If we're dealing with the multilingual setting, then write
                        # a summary for average accuracy across all languages.
                        if isinstance(val_iter, dict):
                            val_tensorboard_logger.scalar_summary(
                                {"val sequence-level accuracy (language average)":
                                 val_seq_acc,
                                 "val perplexity (language average)": val_ppl},
                                self.global_step)
                        for train_scalar_dict, global_step in train_scalar_dicts:
                            train_tensorboard_logger.scalar_summary(
                                train_scalar_dict, global_step)
                        train_scalar_dicts = []

                    # Save a checkpoint each time we run on validation, but only if
                    # there's room in the saved_checkpoints dict.
                    if (save_dir is not None and
                        (len(saved_checkpoints) < max_to_keep or
                         val_seq_acc > min(saved_checkpoints.values()))):
                        checkpoint_path = self.save_to_file(save_dir, run_id)

                        # Update the saved_checkpoints dict by adding the newly saved
                        # checkpoint and taking the top max_to_keep dict items
                        saved_checkpoints[checkpoint_path] = val_seq_acc
                        original_paths = set(saved_checkpoints.keys())
                        if len(saved_checkpoints) > max_to_keep:
                            saved_checkpoints = dict(sorted(
                                saved_checkpoints.items(), key=operator.itemgetter(1),
                                reverse=True)[:max_to_keep])
                        # Get the removed keys, and delete them from disk.
                        for removed_path in (original_paths -
                                             set(saved_checkpoints.keys())):
                            if os.path.exists(removed_path):
                                os.remove(removed_path)

                    # Update the learning rate
                    if self.lr is not None and self.lr_decay is not None:
                        self.optimizer.update_learning_rate(val_ppl, self.epoch_counter)

                # Generator is infinite, so we manually break on each epoch.
                if self.global_step % train_len == 0:
                    break
            if show_progbar:
                epoch_iterator.set_description("Completed {} Epochs".format(
                    self.epoch_counter))
            # If early stopping patience exceeded, stop training
            epoch_val_accs.append(val_seq_acc)
            # If the loss in the patience period show no improvement, stop.
            patience_val_accs = epoch_val_accs[-(early_stopping_patience + 2):]
            if (patience_val_accs == sorted(patience_val_accs, reverse=True) and
                    len(patience_val_accs) == early_stopping_patience + 2):
                # No improvement in patience period, so stop.
                logger.info("Validation sequence accuracies of {} in last {} "
                            "epochs; stopping early due to "
                            "patience of {}.".format(
                                patience_val_accs,
                                early_stopping_patience + 2,
                                early_stopping_patience))
                break
        return saved_checkpoints

    @overrides
    def translate_file(self, src_path, show_progbar=True, n_jobs=1):
        """
        Given a file, predict translations for each data example
        (line of file).

        Parameters
        ----------
        src_path: str
            Path to file with data. The BaseSeq2SeqTranslationModel takes
            three kinds of data --- either OOV data from the original
            dataset, data as generated by the create_word_translation_data
            script, or a file containing just foreign OOVs to translate.

        show_progbar: boolean, optional (default=True)
            Whether or not to show a progress bar.
        """
        sequences_to_translate = []
        num_cols = None
        with open(src_path) as src_file:
            for line in src_file:
                split_line = line.rstrip("\n").split("\t")
                if num_cols is None:
                    num_cols = len(split_line)
                else:
                    assert num_cols == len(split_line)
                # Sequence to translate depends on num columns
                if len(split_line) == 1:
                    # File of just OOVs
                    sequences_to_translate.append(split_line[0])
                elif len(split_line) == 2:
                    # Word translation OOV data with labels
                    sequences_to_translate.append(split_line[0])
                elif len(split_line) == 5:
                    # OOV data as in original dataset
                    sequences_to_translate.append(split_line[0])
                else:
                    raise ValueError(
                        "Unrecognized line format {}".format(split_line))
        # Translate the list of OOV words that we just read.
        pred_translations = self.translate_list(
            sequences_to_translate, show_progbar=show_progbar, n_jobs=n_jobs)
        return pred_translations

    @overrides
    def translate_list(self, src_sequences, show_progbar=True, n_jobs=1, debug=False):
        """
        Given a list of sequences in the source language to
        translate to the target language, run them through the model and
        translate them.

        Parameters
        ----------
        src_sequences: List of str
            A list of str sequences in the source language to
            translate to the target language.

        show_progbar: boolean, optional (default=True)
            Whether or not to show a progress bar during translation.

        Returns
        -------
        target_sequences: List of str
            A list of str with the translation predictions for each sequence
            in src_sequences.
        """
        self.model.eval()
        # Convert the list of src_sequences to a list of Examples.
        src_examples = [data.Example.fromlist(
            [src_sequence], [("source", self.source_field)])
            for src_sequence in src_sequences]
        # Instantiate a Dataset object
        src_dataset = data.Dataset(src_examples, {"source": self.source_field})

        # Create batches on current GPU if CUDA is available, else CPU
        device = None if torch.cuda.is_available() else -1

        # Create an iterator over the source data
        src_iter = data.Iterator(
            dataset=src_dataset, batch_size=self.batch_size, device=device,
            sort=False, repeat=False, shuffle=False, train=False)
        # Run the data through the model to predict translations
        all_predicted_indices = []
        if show_progbar:
            predict_iter = tqdm(
                enumerate(src_iter), total=len(src_iter), file=sys.stdout,
                desc="Prediction Batches")
        else:
            predict_iter = enumerate(src_iter)

        for predict_batch_idx, predict_batch in predict_iter:
            # Sort the source data and lengths by length, and translate it.
            source_data, source_lengths = predict_batch.source
            source_lengths, sort_indices = torch.sort(
                source_lengths, -1, descending=True)
            source_data = Variable(source_data.data.gather(
                1, sort_indices.expand_as(source_data)))

            predicted_indices, _ = self._translate_batch(
                source_data, source_lengths)
            # Reverse the sorting we did for compatibility with the model
            # to restore the original input ordering
            _, reverse_sort_indices = torch.sort(sort_indices, -1)
            predicted_indices = Variable(predicted_indices.data.gather(
                1, reverse_sort_indices.expand_as(predicted_indices)))
            # Originally shape (seq_len, batch_size)
            # Transpose to shape (batch_size, seq_len), and then split into
            # tuple of length batch_size, with each element of shape (1, seq_len)
            all_predicted_indices.extend(predicted_indices.transpose(0, 1).split(1))

        self.model.train()

        # Convert the predicted indices to tokens with the target side vocab
        final_strings = []
        for seq_predicted_indices in all_predicted_indices:
            final_string = []
            for tok_idx in seq_predicted_indices.squeeze(0).data:
                if tok_idx == self.target_padding_idx or tok_idx == self.target_eos_idx:
                    break
                final_string.append(self.target_field.vocab.itos[tok_idx])
            if final_string[-1] == self.target_field.eos_token:
                final_strings.append(self._format_output(final_string[1:-1]))
            else:
                final_strings.append(self._format_output(final_string[1:]))
        return final_strings

    def _build_model(self):
        """
        Create the encoder, decoder, output projection, criterion, and
        optimizer for use in this model. Additionally, move the model to GPU
        if applicable.
        """
        model_loaded = self.model is not None
        # If model has not been created or loaded, create it
        if not model_loaded:
            encoder = EncoderRNN(
                embed_dim=self.embed_dim,
                rnn_dim=self.rnn_dim,
                vocab_size=self.src_vocab_size,
                padding_idx=self.src_padding_idx,
                dropout=self.dropout,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                rnn_type=self.rnn_type)
            decoder = AttentionDecoderRNN(
                embed_dim=self.embed_dim,
                rnn_dim=self.rnn_dim,
                vocab_size=self.target_vocab_size,
                padding_idx=self.target_padding_idx,
                dropout=self.dropout,
                num_layers=self.num_layers,
                input_feed=self.input_feed,
                rnn_type=self.rnn_type)
            output_projection = nn.Sequential(
                nn.Linear(self.rnn_dim, self.target_vocab_size),
                nn.LogSoftmax())
            self.model = Seq2Seq(encoder, decoder)

            # Move model to GPU if is CUDA is available.
            if torch.cuda.is_available():
                self.model.cuda()
                output_projection.cuda()
            else:
                self.model.cpu()
                output_projection.cpu()

            # Set the output projection of the model. This isn't an init
            # argument because there are some parallelism differences between
            # it and the encoder/decoder
            self.model.output_projection = output_projection

        # If the criterion has not been created yet, create it.
        if self.criterion is None:
            self.criterion = NMTCriterion(self.target_vocab_size,
                                          self.target_padding_idx)
        if self.param_init is not None and not model_loaded:
            for p in self.model.parameters():
                p.data.uniform_(-self.param_init, self.param_init)

        # Create optimizer
        if self.optimizer is None:
            self.optimizer = Optimizer(
                method=self.optimizer_str, lr=self.lr,
                max_gradient_norm=self.max_gradient_norm,
                lr_decay=self.lr_decay, start_decay_at=self.start_decay_at)
            self.optimizer.set_parameters(self.model.parameters())
        num_params = sum([p.nelement() for p in self.model.parameters()])
        logger.info("Number of parameters: {}".format(num_params))

    def _calculate_loss(self, outputs, targets, output_projection, criterion,
                        eval=False):
        """Given raw decoder outputs, gold targets, a model to project to
        the output vocabulary dimension, and a crtierion, calculate the
        loss and the number of correct tokens.

        Parameters
        ----------
        outputs: Variable(FloatTensor)
            The raw decoder outputs that will be projected to the vocab_size
            dimension and used to calculate the loss.

        targets: Variable(LongTensor)
            The gold word indices.

        output_projection:
            A model that projects the outputs into the target vocab space
            and normalizes with LogSoftmax.

        criterion:
            The criterion used to calculate the loss.

        Returns
        -------
        loss: int
            The model loss.

        grad_output: Variable(FloatTensor)
            The gradient of the decoder output
            with regards to the loss.

        num_correct: int
            The number of tokens correctly predicted.
        """
        # Compute output projections one timestep at a time to save memory
        num_correct, loss = 0, 0
        outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

        outputs_split = torch.split(outputs, self.output_projection_batch_size)
        targets_split = torch.split(targets, self.output_projection_batch_size)

        batch_size = outputs.size(1)
        for timestep, (output_t, target_t) in enumerate(zip(outputs_split,
                                                            targets_split)):
            # Shape: (batch_size * seq_len)
            flat_target_t = target_t.view(-1)
            # Shape: (batch_size, rnn_dim)
            output_t = output_t.view(-1, output_t.size(2))
            # Shape: (batch_size, target_vocab_size)
            scores_t = output_projection(output_t)

            loss_t = criterion(scores_t, flat_target_t)
            predicted_t = scores_t.max(1)[1]
            non_padding = flat_target_t.ne(self.target_padding_idx).data
            num_correct_t = predicted_t.data.eq(
                flat_target_t.data).masked_select(non_padding).sum()
            num_correct += num_correct_t
            loss += loss_t.data[0]
            if not eval:
                loss_t.div(batch_size).backward()
        grad_output = None if outputs.grad is None else outputs.grad.data
        return loss, grad_output, num_correct

    def _evaluate(self, data_iter, show_progbar=True):
        """
        Given an iterator of validation data, use the model to predict
        and evaluate on it.

        Parameters
        ----------
        data_iter: Iterator
            Iterator that produces validation batches.

        show_progbar: boolean, optional (default=True)
            Whether or not to show a progress bar during training.

        Returns
        -------
        metrics: tuple
            Tuple of (average loss per token, average accuracy per token,
            average sequence exact match accuracy).
        """
        total_loss, total_target_words = 0, 0
        total_sequences = 0
        total_tokens_correct, total_sequences_correct = 0, 0

        self.model.eval()
        if show_progbar:
            eval_iter = tqdm(
                enumerate(data_iter), total=len(data_iter), file=sys.stdout,
                desc="Evaluation Batches")
        else:
            eval_iter = enumerate(data_iter)

        for eval_batch_idx, eval_batch in eval_iter:
            eval_batch_source, eval_batch_target = sort_mt_batch(
                eval_batch.source, eval_batch.target)
            source_data, source_lengths = eval_batch_source
            target_data, target_length = eval_batch_target
            total_sequences += source_data.size(1)

            # Outputs from teacher forcing
            force_outputs = self.model(source_data, source_lengths, target_data)

            # Exclude target init token from targets.
            no_init_target_data = target_data[1:]
            batch_loss, _, batch_num_correct = self._calculate_loss(
                force_outputs, no_init_target_data, self.model.output_projection,
                self.criterion, eval=True)
            total_loss += batch_loss
            total_target_words += no_init_target_data.data.ne(
                self.target_padding_idx).sum()
            total_tokens_correct += batch_num_correct
            # Outputs from decoding
            all_decoded_indices, all_decoded_lengths = self._translate_batch(
                source_data, source_lengths)
            target_data_transposed = target_data.transpose(1, 0)
            for idx, sequence_decoded_indices in enumerate(
                    all_decoded_indices.transpose(1, 0).split(1)):
                # Wrong if the predicted length and the target length aren't the same
                if target_length[idx] != all_decoded_lengths[idx]:
                    continue
                # Squeeze out the 1 dimension created from split, and
                # slice off the length that we predicted.
                sequence_decoded_indices = (sequence_decoded_indices.squeeze(0)[
                    :all_decoded_lengths[idx]].data)

                # Get the corresponding target, and slice off the portion of the target
                # with no padding
                idx_target = target_data_transposed[idx][:target_length[idx]].data
                # Check if the predicted and target indices are equal
                if torch.equal(sequence_decoded_indices, idx_target):
                    total_sequences_correct += 1

        self.model.train()
        return (total_loss / total_target_words,
                total_tokens_correct / total_target_words,
                total_sequences_correct / total_sequences)

    def _format_output(self, output):
        """
        Given a list of strings (the string representations of the
        predicted tokens), process them to produce one output string.

        Parameters
        ----------
        output: List of str
            List of str representing the indices predicted by the model.

        Returns
        -------
        output_str: str
            The predicted tokens represented as a single string.
        """
        raise NotImplementedError

    def _translate_batch(self, source_data, source_lengths, max_sequence_length=20):
        #  Run the source data and the source lengths through the encoder.
        context, enc_hidden, context_lens = self.model.encoder(
            source_data, source_lengths)

        batch_size = context.size(1)
        if isinstance(enc_hidden, tuple):
            decoder_hidden = tuple(self.model._fix_enc_hidden(enc_hidden[i])
                                   for i in range(len(enc_hidden)))
        else:
            decoder_hidden = self.model._fix_enc_hidden(enc_hidden)

        #  Run the decoder to generate sentences, using greedy search
        decoder_output = self.model.make_initial_decoder_output(context)

        # Store the predicted outputs for each batch
        # List of Tensors, where each Tensor is of size (batch_size)
        # and the List will hold the predictions for all batches at each timestep
        predicted_tokens = [Variable(torch.LongTensor([self.target_init_idx] *
                                                      batch_size),
                                     volatile=True)]
        # Store the lengths of each output
        # Tensor of size (batch_size)
        output_lengths = torch.IntTensor(batch_size).zero_()

        if torch.cuda.is_available():
            predicted_tokens[0] = predicted_tokens[0].cuda()
            output_lengths = output_lengths.cuda()
        else:
            predicted_tokens[0] = predicted_tokens[0].cpu()
            output_lengths = output_lengths.cpu()

        for current_time_step in range(max_sequence_length):
            # Take input as last predicted, and add a fake time dimension
            timestep_input = predicted_tokens[-1].unsqueeze(0)
            decoder_output, decoder_hidden, attn = self.model.decoder(
                timestep_input, context, context_lens, decoder_hidden,
                decoder_output)
            decoder_output = decoder_output.squeeze(0)
            output_distribution = self.model.output_projection(decoder_output)

            timestep_predicted_tokens = output_distribution.max(1)[1]
            predicted_tokens.append(timestep_predicted_tokens)

            # Add 2 to sequences due to eos and init tokens, since
            # current_time_step starts from 0
            output_lengths[(predicted_tokens[-1].data == self.target_eos_idx) &
                           (output_lengths == 0)] = current_time_step + 2

            # If we've outputted EOS for everything, just stop.
            if all(output_lengths):
                break
        # Manually add EOS tokens for sentences that are max length.
        eos_to_append = Variable(torch.LongTensor([self.target_eos_idx] *
                                                  batch_size),
                                 volatile=True)
        if torch.cuda.is_available():
            eos_to_append = eos_to_append.cuda()
        else:
            eos_to_append = eos_to_append.cpu()
        predicted_tokens.append(eos_to_append)
        # add 2 to sequences that are max length due to eos and init tokens
        output_lengths[output_lengths == 0] = max_sequence_length + 2
        predicted_tokens = torch.stack(predicted_tokens, 0)
        return predicted_tokens, output_lengths
