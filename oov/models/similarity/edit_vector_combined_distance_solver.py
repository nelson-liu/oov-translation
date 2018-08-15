import torch
import functools
import logging
import math
import multiprocessing
import os
import shutil

from annoy import AnnoyIndex
import dill
import editdistance
from overrides import overrides
import six
from tqdm import tqdm

from .vector_distance_solver import VectorDistanceSolver
from ...utils.fasttext.fasttext import generate_fasttext_vectors_from_list

logger = logging.getLogger(__name__)


def _get_nearest_neighbor_mp_alias(edit_vector_combined_distance_solver,
                                   oov_token_vector_tuple):
    """
    Alias for EditVectorCombinedDistanceSolver._get_nearest_neighbor that
    allows the method to be called in a multiprocessing pool
    """
    return edit_vector_combined_distance_solver._get_nearest_neighbor(
        oov_token_vector_tuple)


class EditVectorCombinedDistanceSolver(VectorDistanceSolver):
    """
    A simple baseline model for doing OOV translation that takes
    the translation of an OOV word to be the translation of the
    in-vocabulary word with the highest interpolation of
    vector similarity + edit similarity.

    We find the word with the highest similarity in the source
    vocabulary, and pick its most likely translation (according to the
    t-table) as our predicted translation.

    We take advantage of the FastText package from Facebook to easily
    generate vectors for unknown words.
    """

    @overrides
    def __init__(self):
        super(EditVectorCombinedDistanceSolver, self).__init__()

        # We don't use self.foreign_vectors, delete to avoid bugs
        del self.foreign_vectors

        self.int_to_foreign = None
        self.annoy_index = None
        self.annoy_index_path = None
        self.vector_dim = None

    @overrides
    def get_state_dict(self):
        state_dict = {
            "solver_class": self.__class__,
            "solver_init_params": self.solver_init_params,
            "fasttext_model_path": self.fasttext_model_path,
            "foreign_to_english": self.foreign_to_english,
            "int_to_foreign": self.int_to_foreign,
            "annoy_index_path": self.annoy_index_path,
            "vector_dim": self.vector_dim
        }
        return state_dict

    @overrides
    def load_from_state_dict(self, state_dict):
        self.fasttext_model_path = state_dict["fasttext_model_path"]
        self.foreign_to_english = state_dict["foreign_to_english"]
        self.int_to_foreign = state_dict["int_to_foreign"]
        self.annoy_index_path = state_dict["annoy_index_path"]
        self.vector_dim = state_dict["vector_dim"]

        self.annoy_index = AnnoyIndex(self.vector_dim)
        self.annoy_index.load(self.annoy_index_path)

        self.was_loaded = True
        return self

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

        # Save the annoy index to the save path
        logger.info(
            "Saving annoy index to save dir at {}".format(save_dir))
        self.annoy_index_path = os.path.join(save_dir,
                                             run_id + "_annoy_index.ann")
        self.annoy_index.save(self.annoy_index_path)

        state_dict = self.get_state_dict()
        torch.save(state_dict, save_path, pickle_module=dill)

    @overrides
    def train_model(self, foreign_vectors, foreign_to_english, num_trees=500,
                    log_dir=None, save_dir=None, run_id=None):
        if self.was_loaded:
            raise ValueError("EditVectorCombinedDistanceSolver does not support "
                             "training from a saved model.")
        # This model has no parameters to optimize
        self.foreign_to_english = foreign_to_english

        # Use FastText to generate vectors for tokens in the
        # foreign_to_english dictionary that aren't in foreign_vectors.
        logger.info("Using FastText to make vectors for tokens that are in "
                    "our foreign to english dictionary, but not in the set "
                    "of pretrained vectors.")
        uncovered_foreign_tokens = [
            tok for tok in self.foreign_to_english if
            tok not in foreign_vectors]
        uncovered_tokens_to_vectors = generate_fasttext_vectors_from_list(
            fasttext_binary_path=self.fasttext_bin_path,
            fasttext_model_path=self.fasttext_model_path,
            input_words=uncovered_foreign_tokens)

        # Add these vectors to the foreign_vectors dict
        for token, vector in uncovered_tokens_to_vectors.items():
            if self.vector_dim is None:
                self.vector_dim = len(vector)
            else:
                assert self.vector_dim == len(vector)
            foreign_vectors[token] = vector

        # Prune the foreign_vectors_dict until the set of foreign tokens in
        # our foreign to english dict is the same as the set of vectors we have
        pruned_foreign_vectors_dict = {
            k: v for k, v in foreign_vectors.items() if
            k in self.foreign_to_english}
        self.int_to_foreign = {
            k: v for k, v in
            enumerate(pruned_foreign_vectors_dict.keys())}

        # Build the annoy index
        logger.info("Building annoy index with {} trees".format(num_trees))
        self.annoy_index = AnnoyIndex(self.vector_dim)
        num_added = 0
        for index, foreign in self.int_to_foreign.items():
            # If we don't have translations for a foreign word, we don't
            # want to propose that as the source for a translation.
            if foreign not in self.foreign_to_english:
                continue
            vector = foreign_vectors[foreign]
            self.annoy_index.add_item(index, vector)
            num_added += 1
        self.annoy_index.build(num_trees)
        assert self.annoy_index.get_n_items() == len(self.foreign_to_english)

        if save_dir is not None and run_id is not None:
            logger.info("Saving trained model to save dir {} with run "
                        "id {}".format(save_dir, run_id))
            self.save_to_file(save_dir=save_dir, run_id=run_id)

    @overrides
    def translate_list(self, oov_list, show_progbar=True, n_jobs=1, debug=False):
        # Get vectors for all of the uncovered_oovs
        oov_vectors = generate_fasttext_vectors_from_list(
            fasttext_binary_path=self.fasttext_bin_path,
            fasttext_model_path=self.fasttext_model_path,
            input_words=oov_list)

        oov_token_candidates_list = []
        num_to_pick = int(math.ceil(0.2 * len(self.foreign_to_english)))
        logger.info("Using annoy to find top {} nearest "
                    "neighbors for each token".format(num_to_pick))
        # Use Annoy to find the top nearest neighbors for each oov token.
        for oov_token in oov_list:
            oov_vector = oov_vectors[oov_token]
            # Find the top 5% of nearest neighbors (in vector space) with the
            # oov token's vector. This tries to find words that are semantically
            # similar.
            nn_indices = self.annoy_index.get_nns_by_vector(
                oov_vector, num_to_pick, search_k=-1, include_distances=False)

            # Get the foreign words corresponding to the found nearest neighbors
            # These are the candidates we will use in the edit distance translation
            candidate_foreigns = [
                self.int_to_foreign[index] for index in nn_indices]
            oov_token_candidates_list.append((oov_token, candidate_foreigns))

        if n_jobs > 1:
            # Since we can't pickle self.annoy_index, set it to a local variable
            # and then delete it.
            annoy_index = self.annoy_index
            del self.annoy_index
            logger.info("Translating with {} processes".format(n_jobs))
            pool = multiprocessing.Pool(processes=n_jobs)
            if six.PY2:
                # Create a multiprocess pool with the _get_nearest_neighbor alias.
                # This is not used in python 3 because there's overhead in passing
                # the object back and forth.
                _bound_get_nearest_neighbor_mp_alias = functools.partial(
                    _get_nearest_neighbor_mp_alias, self)
                closest_source_tokens = pool.map(_bound_get_nearest_neighbor_mp_alias,
                                                 oov_token_candidates_list)
            else:
                closest_source_tokens = pool.map(self._get_nearest_neighbor,
                                                 oov_token_candidates_list)
            # Restore self.annoy_index
            self.annoy_index = annoy_index
        else:
            if show_progbar:
                oov_iterable = tqdm(oov_token_candidates_list)
            else:
                oov_iterable = oov_token_candidates_list
            closest_source_tokens = [
                self._get_nearest_neighbor(oov_token_vector_tuple)
                for oov_token_vector_tuple in oov_iterable]
        predicted_translations = []
        for source_token in closest_source_tokens:
            english_translations = self.foreign_to_english[source_token]
            predicted_translation = max(english_translations.keys(),
                                        key=lambda k: english_translations[k])
            predicted_translations.append(predicted_translation)
        return predicted_translations

    def _get_nearest_neighbor(self, oov_token_candidates_tuple):
        """
        Given a single OOV token, find the best English translation.

        Parameters
        ----------
        oov_token_candidates_tuple: tuple of (str, List[str])
            Tuple of (oov_token, candidates). The oov token is the string
            to predict a translation for. Candidates are the words
            we can choose among as potential source words for translation.
        """
        oov_token, foreign_candidates = oov_token_candidates_tuple

        # Out of the candidates, pick the one with the highest
        # edit similarity.
        def calculate_edit_similarity_with_input_oov(x):
            if len(x) == 0:
                return 0
            longest_common_prefix_len = len(
                os.path.commonprefix([x, oov_token]))
            edit_distance = int(editdistance.eval(x, oov_token))
            score = (0.75 * (1 - (edit_distance /
                                  max(len(x), len(oov_token)))) +
                     0.25 * (longest_common_prefix_len /
                             min(len(x), len(oov_token))))
            return score

        most_similar_source_token = max(
            foreign_candidates, key=calculate_edit_similarity_with_input_oov)
        return most_similar_source_token
