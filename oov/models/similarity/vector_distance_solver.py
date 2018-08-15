import torch
import functools
import logging
import multiprocessing
import os
import shutil

import dill
import numpy as np
from overrides import overrides
from scipy.spatial.distance import cdist
import six
from tqdm import tqdm

from ..base_solver import BaseSolver
from ...utils.fasttext.fasttext import generate_fasttext_vectors_from_list
from ...utils.general.data import get_num_lines

logger = logging.getLogger(__name__)

# global variables to make multiprocessing faster
source_word_vectors = None
source_words = None


def _get_nearest_neighbor_mp_alias(vector_distance_solver, oov_token_and_vector):
    """
    Alias for VectorDistanceSolver._get_nearest_neighbor that allows the
    method to be called in a multiprocessing pool
    """
    return vector_distance_solver._get_nearest_neighbor(oov_token_and_vector)


class VectorDistanceSolver(BaseSolver):
    """
    A simple baseline model for doing OOV translation that takes
    the translation of an OOV word to be the translation of the
    in-vocabulary word with the closest vector distance.

    We find the word with the highest vector similarity in the source
    vocabulary, and pick its most likely translation (according to the
    t-table) as our predicted translation.

    We take advantage of the FastText package from Facebook to easily
    generate vectors for unknown words.

    Attributes
    ----------
    fasttext_bin_path:
        The path to the compiled FastText binary in the repo submodule.

    fasttext_model_path: str
        The path to the FastText model for the language.

    foreign_to_english: Dict of str to Dict of str
        A dict mapping foreign strings to dictionaries containing
        English translations as the keys and weights as the values

    foreign_vectors: Dict of str to ndarray:
        A dict mapping foreign string to their vectors.
    """

    def __init__(self):
        self.solver_init_params = locals()
        self.solver_init_params.pop("self")
        self.solver_init_params.pop("__class__", None)

        self.fasttext_bin_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir,
            os.pardir, "util", "fastText", "fasttext"))
        if not os.path.exists(self.fasttext_bin_path):
            raise OSError(
                "FastText binary path at {} does not exist, "
                "have you compiled it?".format(self.fasttext_bin_path))

        self.fasttext_model_path = None
        self.foreign_to_english = None
        self.foreign_vectors = None
        self.was_loaded = False

    @overrides
    def get_state_dict(self):
        state_dict = {
            "solver_class": self.__class__,
            "solver_init_params": self.solver_init_params,
            "fasttext_model_path": self.fasttext_model_path,
            "foreign_vectors": self.foreign_vectors,
            "foreign_to_english": self.foreign_to_english
        }
        return state_dict

    @overrides
    def load_from_state_dict(self, state_dict):
        self.fasttext_model_path = state_dict["fasttext_model_path"]
        self.foreign_vectors = state_dict["foreign_vectors"]
        self.foreign_to_english = state_dict["foreign_to_english"]
        self.was_loaded = True
        return self

    @overrides
    def load_from_file(self, load_path):
        state_dict = torch.load(load_path)
        return self.load_from_state_dict(state_dict)

    @overrides
    def read_data(self, tgt_given_src_path, fasttext_vectors_path,
                  fasttext_model_path):
        """
        Read the tgt_given_src t-table to get a mapping from foreign
        word to possible English translations.

        Parameters
        -----------
        tgt_given_src_path: str
            Path to t-table with probabilities of target (English)
            tokens given a source token. The expected format is:
            <english token><space><foreign token><space><weight>

        fasttext_vectors_path:
            Pretrained vectors from training a FastText model on the language.

        fasttext_model_path:
            Binary model from training a FastText model on the language.

        Returns
        -------
        data_dict: str
            A dictionary of kwargs to be passed to the train_model
            function.
        """
        if self.was_loaded:
            raise ValueError("VectorDistanceSolver does not support "
                             "training from a saved model.")
        foreign_to_english = {}
        with open(tgt_given_src_path) as tgt_given_src_file:
            num_skipped = 0
            for line in tqdm(tgt_given_src_file,
                             total=get_num_lines(tgt_given_src_path)):
                split_line = line.rstrip("\n").split(" ")
                if len(split_line) == 3:
                    target, source, weight = split_line
                else:
                    num_skipped += 1
                    continue
                if source not in foreign_to_english:
                    foreign_to_english[source] = {}
                foreign_to_english[source][target] = float(weight)
            if num_skipped != 0:
                logger.warning("Skipped {} lines in tgt_given_src due to improper "
                               "whitespace format.".format(num_skipped))

        # Read the FastText vectors into a dictionary of string to numpy array
        logger.info("Reading FastText vectors from {}".format(fasttext_vectors_path))
        foreign_vectors = {}
        with open(fasttext_vectors_path) as fasttext_vectors_file:
            # Skip the header
            next(fasttext_vectors_file)
            for line in tqdm(fasttext_vectors_file,
                             total=get_num_lines(fasttext_vectors_path)):
                split_line = line.rstrip().split(" ")
                word = split_line[0]
                vector = [float(i) for i in split_line[1:]]
                foreign_vectors[word] = np.array(vector)

        self.fasttext_model_path = os.path.normpath(fasttext_model_path)
        return {"foreign_vectors": foreign_vectors,
                "foreign_to_english": foreign_to_english}

    @overrides
    def save_to_file(self, save_dir, run_id):
        save_path = os.path.join(save_dir, run_id + "_model.pkl")
        # Move the fastText model we used to the save path
        logger.info(
            "Copying fastText model from {} to "
            "save dir at {}".format(self.fasttext_model_path, save_dir))
        shutil.copy(self.fasttext_model_path, save_dir)
        # Now edit the model path to point to file we wrote
        self.fasttext_model_path = os.path.join(
            save_dir, os.path.basename(self.fasttext_model_path))
        state_dict = self.get_state_dict()
        torch.save(state_dict, save_path, pickle_module=dill)

    @overrides
    def train_model(self, foreign_vectors, foreign_to_english, log_dir=None,
                    save_dir=None, run_id=None):
        if self.was_loaded:
            raise ValueError("VectorDistanceSolver does not support "
                             "training from a saved model.")
        # This model has no parameters to optimize
        self.foreign_vectors = foreign_vectors
        self.foreign_to_english = foreign_to_english

        # Use FastText to generate vectors for tokens in the
        # foreign_to_english dictionary that aren't in foreign_vectors.
        uncovered_foreign_tokens = [
            tok for tok in self.foreign_to_english if
            tok not in self.foreign_vectors]
        uncovered_tokens_to_vectors = generate_fasttext_vectors_from_list(
            fasttext_binary_path=self.fasttext_bin_path,
            fasttext_model_path=self.fasttext_model_path,
            input_words=uncovered_foreign_tokens)

        # Add these vectors to the foreign_vectors dict
        for token, vector in uncovered_tokens_to_vectors.items():
            self.foreign_vectors[token] = vector

        if save_dir is not None and run_id is not None:
            logger.info("Saving trained model to save dir {} with run "
                        "id {}".format(save_dir, run_id))
            self.save_to_file(save_dir=save_dir, run_id=run_id)

    @overrides
    def translate_file(self, oov_path, show_progbar=True, n_jobs=1):
        """
        Given a file, predict translations for each data example
        (line of file).

        Parameters
        ----------
        oov_path: str
            Path to file with data. The VectorDistanceSolver takes
            three kinds of data --- either vanilla OOV data as in the
            original dataset dev and test splits, OOV data as generated
            by the create_word_translation_data script, or a file containing
            just foreign OOVs to translate.
        """
        sequences_to_translate = []
        num_cols = None
        with open(oov_path) as src_file:
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
    def translate_list(self, oov_list, show_progbar=True, n_jobs=1, debug=False):
        oov_vectors = {oov: self.foreign_vectors[oov] for oov in oov_list if
                       oov in self.foreign_vectors}
        uncovered_oovs = [oov for oov in oov_list if oov not in
                          self.foreign_vectors]

        # Get vectors for all of the uncovered_oovs
        uncovered_oovs_to_vectors = generate_fasttext_vectors_from_list(
            fasttext_binary_path=self.fasttext_bin_path,
            fasttext_model_path=self.fasttext_model_path,
            input_words=uncovered_oovs)

        # Add the generated vectors back to the dictionary
        for oov, generated_vector in uncovered_oovs_to_vectors.items():
            oov_vectors[oov] = generated_vector

        global source_words
        source_words = list(self.foreign_to_english.keys())

        # Shape: (num source words, vector dimension)
        global source_word_vectors
        source_word_vectors = np.array([self.foreign_vectors[source_word] for
                                        source_word in source_words])

        if n_jobs > 1:
            logger.info("Translating with {} processes".format(n_jobs))
            pool = multiprocessing.Pool(processes=n_jobs)
            oov_token_vector_list = [(oov, oov_vectors[oov]) for oov in oov_list]
            if six.PY2:
                # Create a multiprocess pool with the _get_nearest_neighbor alias.
                # This is not used in python 3 because there's overhead in passing
                # the object back and forth.
                _bound_get_nearest_neighbor_mp_alias = functools.partial(
                    _get_nearest_neighbor_mp_alias, self)
                closest_source_tokens = pool.map(_bound_get_nearest_neighbor_mp_alias,
                                                 oov_token_vector_list)
            else:
                closest_source_tokens = pool.map(
                    self._get_nearest_neighbor, oov_token_vector_list)
        else:
            if show_progbar:
                oov_iterable = tqdm(oov_list)
            else:
                oov_iterable = oov_list
            closest_source_tokens = [self._get_nearest_neighbor((oov, oov_vectors[oov]))
                                     for oov in oov_iterable]
        predicted_translations = []
        for source_token in closest_source_tokens:
            english_translations = self.foreign_to_english[source_token]
            predicted_translation = max(english_translations.keys(),
                                        key=lambda k: english_translations[k])
            predicted_translations.append(predicted_translation)
        return predicted_translations

    def _get_nearest_neighbor(self, oov_token_and_vector):
        """
        Given a single OOV token, find the source token that is most similar.

        Parameters
        ----------
        oov_token_and_vector: tuple of (str, ndarray)
            A tuple where the first item is the string oov token to predict a
            translation for and the second item is the vector of the oov token
            to predict a translation for.
        """
        oov_token, oov_vector = oov_token_and_vector
        # Shape: (1, vector dimension)
        oov_vector = oov_vector.reshape(1, -1)

        # Find the string in the vocab that has the highest similarity
        # with the OOV token.
        # Get the cosine similarity of the oov token with all the source word vectors

        # cdist output is shape (1, num_source_words)
        most_similar_source_index = np.argmax(1 - cdist(
            oov_vector, source_word_vectors, "cosine"))
        most_similar_source_token = source_words[most_similar_source_index]
        return most_similar_source_token
