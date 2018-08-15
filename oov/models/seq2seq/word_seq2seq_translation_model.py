from overrides import overrides
from torchtext import data

from .base_seq2seq_translation_model import BaseSeq2SeqTranslationModel


class WordSeq2SeqTranslationModel(BaseSeq2SeqTranslationModel):
    """
    This model is a standard word-level sequence-to-sequence model for
    translating between languages.

    To train, we pass the model a sequence of words in the source language
    and the model is tasked with predicting a sequence of words in the target
    language. The training files for WMT EN/DE are pretokenized with
    scripts/data/tokenize_en_de.py

    At test time, we are shown an foreign sentence to translate as simply a sequence
    of words, and we accordingly predict its target translation as a sequence of
    words.
    """

    @overrides
    def read_data(self, train_file, val_file, max_source_vocab=50000,
                  max_target_vocab=50000):
        if self.source_field is None:
            self.source_field = data.Field(include_lengths=True)
        if self.target_field is None:
            self.target_field = data.Field(include_lengths=True,
                                           init_token="<bos>",
                                           eos_token="<eos>")
        return super(WordSeq2SeqTranslationModel, self).read_data(
            train_file=train_file,
            val_file=val_file,
            max_source_vocab=max_source_vocab,
            max_target_vocab=max_target_vocab)

    @overrides
    def _format_output(self, output):
        return " ".join(output)
