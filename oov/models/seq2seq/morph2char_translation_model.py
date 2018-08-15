import logging
import re

from overrides import overrides
from torchtext import data
from tqdm import tqdm

from ...utils.general.data import get_num_lines
from .base_seq2seq_translation_model import BaseSeq2SeqTranslationModel

logger = logging.getLogger(__name__)


class Morph2CharTranslationModel(BaseSeq2SeqTranslationModel):
    """
    This model is a sequence-to-sequence model for
    translating between languages. The encoder is a morpheme-level encoder,
    and the decoder is a character-level decoder.

    To train, we pass the model a sequence of morphemes in the source language
    and the model is tasked with predicting a sequence of characters in the
    target language. The training files are created with
    scripts/data/create_word_translation_data.py

    This model was created with the task of translation of individual words
    between languages in mind. At test time, we are shown an OOV word to
    translate as simply a sequence of morphemes, and we accordingly predict
    its target translation as a sequence of characters.
    """
    @overrides
    def get_state_dict(self):
        state_dict = super(Morph2CharTranslationModel, self).get_state_dict()
        state_dict["src_to_morphemes"] = self.src_to_morphemes
        return state_dict

    @overrides
    def load_from_state_dict(self, state_dict):
        self.src_to_morphemes = state_dict["src_to_morphemes"]
        return super(Morph2CharTranslationModel,
                     self).load_from_state_dict(state_dict)

    @overrides
    def read_data(self, train_file, val_file, src_to_morpheme_path,
                  max_source_vocab=50000, max_target_vocab=50000):
        """
        Read data for the morpheme seq2seq model. See base class
        read_data docstring for more information.

        Parameters
        ----------
        train_file: str
            The string path to the file with the train data.

        val_file: str
            The string path to the file with the validation data.

        src_to_morpheme_path: str
            The string path to the file with morpheme segmentations
            for the source side.

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
        if (not hasattr(self, "src_to_morphemes") or not
                self.src_to_morphemes):
            logger.info("Reading src_to_morphemes segmentations "
                        "from {}".format(src_to_morpheme_path))
            src_to_morphemes = {}
            with open(src_to_morpheme_path) as src_to_morpheme_file:
                for line in tqdm(src_to_morpheme_file,
                                 total=get_num_lines(src_to_morpheme_path)):
                    original_word, segmentation = line.rstrip("\n").split("\t")[:2]
                    src_to_morphemes[original_word] = segmentation.split(" ")
            self.src_to_morphemes = src_to_morphemes

        if self.source_field is None:
            self.source_field = data.Field(tokenize=self.morphemes_split_src_word,
                                           include_lengths=True)
        if self.target_field is None:
            self.target_field = data.Field(tokenize=list,
                                           include_lengths=True,
                                           init_token="<bos>",
                                           eos_token="<eos>")
        return super(Morph2CharTranslationModel, self).read_data(
            train_file=train_file,
            val_file=val_file,
            max_source_vocab=max_source_vocab,
            max_target_vocab=max_target_vocab)

    def morphemes_split_src_word(self, input_string):
        """
        Given an input source foreign string, segment it into morphemes.
        We first split tokens on space to get words, then further segment
        those words into morphemes if possible. If the input string is not
        in the dictionary, we return a list with the input_string.

        Parameters
        ----------
        input_string: str
            The string to break into morphemes.

        Returns
        -------
        chunked_string: List of str
             A list of strings representing the constituent morphemes
             of the character.
        """
        morphemes = []
        for word in re.split(r'(\s+)', input_string):
            morphemes.extend(self.src_to_morphemes.get(word, list(word)))
        return morphemes

    @overrides
    def _format_output(self, output):
        return "".join(output)
