from __future__ import division
import torch
from torch.autograd import Variable
from torchtext import data

import logging
import os
import sys

import dill
from overrides import overrides
from tqdm import tqdm


from ..base_solver import BaseSolver
from .base_seq2seq_translation_model import BaseSeq2SeqTranslationModel

logger = logging.getLogger(__name__)


class Seq2SeqEnsembleTranslationModel(BaseSolver):
    """
    A model that wraps other Seq2Seq models and uses them in an ensemble.

    When decoding, the probability distribution over the output is taken to be
    the average of the output probability distribution of each model in the
    ensemble.
    """
    def __init__(self, seq2seq_models=None):
        """
        Create a new Seq2SeqEnsembleTranslationModel.

        Parameters
        ----------
        seq2seq_models: dict of str to BaseSeq2SeqTranslationModel or str, optional
            A dictionary of str model_ids to BaseSeq2SeqTranslationModel
            objects or strings representing paths to load
            BaseSeq2SeqTranslationModel subclasses from.
        """
        self.solver_init_params = locals()
        self.solver_init_params.pop("self")
        self.solver_init_params.pop("__class__", None)
        self.solver_init_params.pop("seq2seq_models")

        # These data members must be the same across the models we are ensembling.
        self.batch_size = None
        self.target_init_idx = None
        self.target_padding_idx = None
        self.target_eos_idx = None
        self.source_field = None
        self.target_field = None

        self.models = {}
        if seq2seq_models is not None:
            if not isinstance(seq2seq_models, dict):
                raise ValueError("Expected seq2seq_models to be a dict, "
                                 "but got {}".format(type(seq2seq_models)))
            for model_id, model in seq2seq_models.items():
                if isinstance(model, BaseSeq2SeqTranslationModel):
                    # Add this model directly to the ensemble.
                    self.models[model_id] = model
                elif isinstance(model, str):
                    try:
                        # Load a BaseSeq2SeqTranslationModel subclass from the str path.
                        load_path = model
                        logger.info("Trying to load model with "
                                    "id {} at {}".format(model_id, load_path))
                        if torch.cuda.is_available():
                            model_state_dict = torch.load(load_path)
                        else:
                            model_state_dict = torch.load(
                                load_path, map_location=lambda storage, loc: storage)
                    except IOError:
                        # load_path does not exist on filesystem, so skip it.
                        # We might load this same model from state dict, so this is only
                        # really a concern if you're training a model.
                        logger.warning(
                            "Tried to load model at {} while creating ensemble, "
                            "but path doesn't exist. Maybe we'll load this model "
                            "from the state dict. This is only really a concern "
                            "if you're trying to train the ensemble".format(load_path))
                        continue
                    # Construct a copy of the saved model.
                    loaded_model = model_state_dict["solver_class"](
                        **model_state_dict["solver_init_params"])

                    # Load the constructed copy's state.
                    loaded_model = loaded_model.load_from_state_dict(
                        model_state_dict)

                    # Add it to the dict of models.
                    self.models[model_id] = loaded_model
                else:
                    raise ValueError(
                        "Values of models dict is expected to be "
                        "BaseSeq2SeqTranslationModel "
                        "subclass or str, got type {} for key {}".format(
                            type(model), model_id))

                # Set and check batch size
                if self.batch_size is None:
                    self.batch_size = self.models[model_id].batch_size
                elif self.batch_size != self.models[model_id].batch_size:
                    raise ValueError("Models have different batch sizes.")

                # Set and check data members
                if self.target_init_idx is None:
                    self.target_init_idx = self.models[model_id].target_init_idx
                    self.target_padding_idx = self.models[model_id].target_padding_idx
                    self.target_eos_idx = self.models[model_id].target_eos_idx
                    self.source_field = self.models[model_id].source_field
                    self.target_field = self.models[model_id].target_field
                elif (self.target_init_idx != self.models[model_id].target_init_idx or
                      self.target_padding_idx != self.models[
                          model_id].target_padding_idx or
                      self.target_eos_idx != self.models[model_id].target_eos_idx):
                    raise ValueError("Models have different target init, padding, or "
                                     "eos tokens.")

    def add_model(self, model, model_id):
        """
        Given a model and an id for the model, add it to the model combination.
        Parameters
        ----------
        model: BaseSeq2SeqTranslationModel
            The model to add to the combination. Must inherit from
            BaseSeq2SeqTranslationModel and implement the BaseModel public API.

        model_id: str
            The string to associate with this model. If a model in the
            combination already has this model_id, an error will be raised.
        """
        if model_id in self.models:
            raise ValueError("Input model_id {} is already in the collection "
                             "of models ({})".format(model_id, list(model_id.keys())))
        if not isinstance(model, BaseSeq2SeqTranslationModel):
            raise ValueError("Input model {} does not subclass "
                             "BaseSeq2SeqTranslationModel".format(model))
        if model.batch_size != self.batch_size:
            raise ValueError("Input model {} has batch size {}, "
                             "ensemble batch size is {}".format(model.batch_size,
                                                                self.batch_size))
        self.models[model_id] = model

    @overrides
    def get_state_dict(self):
        """
        Returns
        -------
        state_dict: dict
            Dict containing values of members and other model-specific
            things to be saved.
        """
        model_state_dicts = {}

        # Get state dict for each model
        for model_id, model in self.models.items():
            model_state_dicts[model_id] = model.get_state_dict()

        state_dict = {
            "solver_class": self.__class__,
            "solver_init_params": self.solver_init_params,
            "models": model_state_dicts,
            "batch_size": self.batch_size,
            "target_init_idx": self.target_init_idx,
            "target_padding_idx": self.target_padding_idx,
            "target_eos_idx": self.target_eos_idx,
            "target_field": self.target_field,
            "source_field": self.source_field,
        }
        return state_dict

    @overrides
    def load_from_state_dict(self, state_dict):
        if self.models != {}:
            logger.warning("Loading a state dict, but this ensemble already "
                           "has constituent models! Removing existing constituent "
                           "models and using models loaded from state dict instead.")
            self.models = {}
        # Instantiate the models from their saved state dictionaries
        for model_id, model_state_dict in state_dict["models"].items():
            # Construct a copy of the saved model.
            loaded_model = model_state_dict["solver_class"](
                **model_state_dict["solver_init_params"])

            # Load the constructed copy's state
            loaded_model = loaded_model.load_from_state_dict(model_state_dict)

            self.models[model_id] = loaded_model

        self.batch_size = state_dict["batch_size"]
        self.target_init_idx = state_dict["target_init_idx"]
        self.target_padding_idx = state_dict["target_padding_idx"]
        self.target_eos_idx = state_dict["target_eos_idx"]
        self.source_field = state_dict["source_field"]
        self.target_field = state_dict["target_field"]
        return self

    @overrides
    def load_from_file(self, checkpoint_path):
        logger.info("Loading models from checkpoint "
                    "at {}".format(checkpoint_path))
        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage)
        return self.load_from_state_dict(state_dict)

    @overrides
    def read_data(self):
        """
        This model has no data to read.

        Returns
        -------
        data: dict
            An empty dict, since the train_model does not need any data as well.
        """
        return {}

    @overrides
    def save_to_file(self, save_dir, run_id):
        """
        Save the state of the SolverCombination object to a file.

        Parameters
        ----------
        save_dir: str
            The folder to save the trained model in.

        run_id: str
            The run_id for this model, also dictates the serialization filename.
        """
        save_path = os.path.join(save_dir, run_id + "_model.pkl")
        state_dict = self.get_state_dict()
        torch.save(state_dict, save_path, pickle_module=dill)

    @overrides
    def train_model(self, log_dir=None, save_dir=None, run_id=None):
        """
        "Train" the model (this is a no-op), and then save the ensemble to disk.

        Parameters
        ----------

        log_dir: str, optional (default=None)
            The folder to which to save logs. We create separate train and val
            folders in this directory to hold train and validation logs.

        save_dir: str, optional (default=None)
            The folder to save model checkpoints to. If this is provided, you
            must also provide a value for run_id.

        run_id: str, optional (default=None)
            The run_id for this experiment. This is used to identify the
            checkpoint files.
        """
        # If save_dir is provided, run_id must be provided as well
        if save_dir is not None and run_id is None:
            raise ValueError("Must pass in run_id if you "
                             "wish to save checkpoints.")
        if save_dir is not None and run_id is not None:
            logger.info("Saving trained model to save dir {} with run "
                        "id {}".format(save_dir, run_id))
            self.save_to_file(save_dir=save_dir, run_id=run_id)

    @overrides
    def translate_file(self, src_path, show_progbar=True, n_jobs=1):
        """
        Given a file, predict translations for each data example
        (line of file).

        Parameters
        ----------
        src_path: str
            Path to file with data. The Seq2SeqEnsembleTranslationModel takes
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
            sequences_to_translate, show_progbar=show_progbar)
        return pred_translations

    @overrides
    def translate_list(self, src_sequences, show_progbar=True, n_jobs=1, debug=False):
        """
        Given a list of sequences in the source language to
        translate to the target language, run them through the ensemble and
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
        for model_id in self.models:
            self.models[model_id].model.eval()

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

        for model_id in self.models:
            self.models[model_id].model.train()

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
        return list(self.models.values())[0]._format_output(output)

    def _translate_batch(self, source_data, source_lengths, max_sequence_length=20):
        # Run the source data and the source lengths through each of the model encoders.
        encoded_source = {model_id: model.model.encoder(source_data, source_lengths) for
                          model_id, model in self.models.items()}

        # Assert the output batch size is the same for all models
        batch_sizes = [encoded_tuple[0].size(1) for model_id, encoded_tuple
                       in encoded_source.items()]
        if len(set(batch_sizes)) > 1:
            raise ValueError("Got batch sizes {} from models. "
                             "Batch sizes must be the same".format(batch_sizes))
        batch_size = batch_sizes[0]

        # Dict to store decoder inputs for each model
        decoder_inputs = {}

        # Get initial decoder hidden state for each model from the encoder hidden state.
        for model_id, encoded_tuple in encoded_source.items():
            enc_hidden = encoded_tuple[1]
            model = self.models[model_id]
            if isinstance(enc_hidden, tuple):
                decoder_hidden = tuple(model.model._fix_enc_hidden(enc_hidden[i])
                                       for i in range(len(enc_hidden)))
            else:
                decoder_hidden = model.model._fix_enc_hidden(enc_hidden)
            if model_id not in decoder_inputs:
                decoder_inputs[model_id] = {}
            decoder_inputs[model_id]["decoder_hidden"] = decoder_hidden

        # Get the initial decoder output for each model
        for model_id, encoded_tuple in encoded_source.items():
            context = encoded_tuple[0]
            model = self.models[model_id]
            decoder_inputs[model_id][
                "decoder_output"] = model.model.make_initial_decoder_output(context)

        # Store the ensemble predicted outputs for each batch
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

        # Iterate over the timesteps
        for current_time_step in range(max_sequence_length):
            # Take input as last predicted, and add a fake time dimension
            timestep_input = predicted_tokens[-1].unsqueeze(0)

            # Run the decoder of each model, and store the output distributions
            output_distributions = []
            for model_id, model in self.models.items():
                context = encoded_source[model_id][0]
                context_lens = encoded_source[model_id][2]
                decoder_hidden = decoder_inputs[model_id]["decoder_hidden"]
                decoder_output = decoder_inputs[model_id]["decoder_output"]

                decoder_output, decoder_hidden, _ = model.model.decoder(
                    timestep_input, context, context_lens, decoder_hidden,
                    decoder_output)

                decoder_output = decoder_output.squeeze(0)
                # Update the decoder output and decoder hidden in our dict
                decoder_inputs[model_id]["decoder_hidden"] = decoder_hidden
                decoder_inputs[model_id]["decoder_output"] = decoder_output

                output_distributions.append(model.model.output_projection(decoder_output))

            output_distribution = torch.mean(torch.stack(output_distributions),
                                             dim=0).squeeze(0)
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

        # Add 2 to sequences that are max length due to eos and init tokens
        output_lengths[output_lengths == 0] = max_sequence_length + 2
        predicted_tokens = torch.stack(predicted_tokens, 0)
        return predicted_tokens, output_lengths
