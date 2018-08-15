import torch
import logging
import pprint

from overrides import overrides
from torchtext import data
from tqdm import tqdm

from .base_seq2seq_translation_model import BaseSeq2SeqTranslationModel

logger = logging.getLogger(__name__)


class MultilingualCharSeq2SeqTranslationModel(BaseSeq2SeqTranslationModel):
    """
    This model is a character-level sequence-to-sequence model for translating
    between languages, but it uses a shared encoder across multiple languages.

    The main difference, functionally, between this model and passing a concatenated
    dataset to CharSeq2SeqTranslationModel is that in this model, we balance the batches
    so each batch has proportionally an appropriate amount of samples for each language.
    As a reult, the amount of iterations it takes to complete an epoch for each language
    is approximately the same.

    ###########################################################################
    # The input data to this model must be romanized, or there's little point #
    # in sharing parameters across the languages!                             #
    ###########################################################################

    To train, we pass the model a sequence of characters in the source languages
    and the model is tasked with predicting a sequence of characters in the target
    language. The training files are created with
    scripts/data/create_word_translation_data.py

    This model was created with the task of translation of individual words between
    languages in mind. At test time, we are shown an OOV word to translate as simply
    a sequence of characters, and we accordingly predict its target translation as a
    sequence of characters.
    """

    @overrides
    def read_data(self, train_files, val_files, max_source_vocab=50000,
                  max_target_vocab=50000):
        if not (isinstance(train_files, dict) and isinstance(val_files, dict)):
            raise ValueError("train_files has type {}, and val_files has type {}. "
                             "Expected dictionaries of language id to data file "
                             "id for both".format(type(train_files), type(val_files)))
        # These fields will be used across all datasets, so they
        # share the same vocabulary.
        if self.source_field is None:
            self.source_field = data.Field(tokenize=list,
                                           include_lengths=True)
        if self.target_field is None:
            self.target_field = data.Field(tokenize=list,
                                           include_lengths=True,
                                           init_token="<bos>",
                                           eos_token="<eos>")
        # For each input language and associated data file, create a TabularDataset
        logger.info("Creating {} Datasets for input train files".format(len(train_files)))
        train_datasets = {}
        for lang, train_path in tqdm(train_files.items()):
            train_datasets[lang] = data.TabularDataset(
                path=train_path, format="tsv",
                fields=[("source", self.source_field),
                        ("target", self.target_field)])
        logger.info("Creating {} Datasets for input val files".format(len(val_files)))
        val_datasets = {}
        for lang, val_path in tqdm(val_files.items()):
            val_datasets[lang] = data.TabularDataset(
                path=val_path, format="tsv",
                fields=[("source", self.source_field),
                        ("target", self.target_field)])

        # Build the vocabulary across all the train datasets
        logger.info("Building source vocabulary.")
        if not hasattr(self.source_field, "vocab"):
            self.source_field.build_vocab(*train_datasets.values(),
                                          max_size=max_source_vocab)
        logger.info("Building target vocabulary.")
        if not hasattr(self.target_field, "vocab"):
            self.target_field.build_vocab(*train_datasets.values(),
                                          max_size=max_target_vocab)
        # Create batches on current GPU if CUDA is available, else CPU
        device = None if torch.cuda.is_available() else -1

        # Get the proper batch size for each language, since we want
        # the batches to have the same proportion of each language across each epoch
        logger.info("Calculating batch proportions for each language.")
        train_dataset_sizes = {lang: len(dataset) for lang, dataset in
                               train_datasets.items()}
        train_batch_sizes = {}
        num_train_examples = sum(train_dataset_sizes.values())
        remainders = {}
        for lang, train_dataset_size in train_dataset_sizes.items():
            # Floor (int) division
            train_batch_sizes[lang] = ((train_dataset_size * self.batch_size) //
                                       num_train_examples)
            if train_batch_sizes[lang] == 0:
                train_batch_sizes[lang] = 1
                remainders[lang] = 0
            else:
                remainders[lang] = (((train_dataset_size * self.batch_size) %
                                    num_train_examples) // num_train_examples)

        # Check how far off we are from the desired batch size (k), and increment the
        # languages with the top k remainders
        k = max(self.batch_size - sum(train_batch_sizes.values()), 0)
        for lang, _ in sorted(remainders.items(), key=lambda x: x[1], reverse=True)[:k]:
            train_batch_sizes[lang] += 1

        assert self.batch_size == sum(train_batch_sizes.values())
        logger.info("Batch sizes for each language: ")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(train_batch_sizes)

        # Create iterators for each of the train and val datasets
        logger.info("Creating iterators for train datasets.")
        train_iters = {}
        for lang, train_dataset in tqdm(train_datasets.items()):
            batch_size = train_batch_sizes[lang]
            train_iter = data.BucketIterator(
                dataset=train_dataset, batch_size=batch_size, device=device,
                sort_key=lambda x: data.interleave_keys(len(x.source),
                                                        len(x.target)))
            train_iters[lang] = train_iter
        logger.info("Creating iterators for val datasets.")
        val_iters = {}
        for lang, val_dataset in tqdm(val_datasets.items()):
            batch_size = self.batch_size
            val_iter = data.BucketIterator(
                dataset=val_dataset, batch_size=batch_size, device=device, train=False,
                sort_key=lambda x: data.interleave_keys(len(x.source),
                                                        len(x.target)))
            val_iters[lang] = val_iter
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
        return {"train_iter": train_iters, "val_iter": val_iters}

    @overrides
    def _format_output(self, output):
        return "".join(output)
