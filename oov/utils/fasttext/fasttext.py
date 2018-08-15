import logging
import os
import subprocess
import tempfile

import numpy as np

logger = logging.getLogger(__name__)


def generate_fasttext_vectors_from_list(fasttext_binary_path,
                                        input_words,
                                        fasttext_model_path=None,
                                        verbose=True):
    """
    Given the path to a binary FastText model and a list of input words to get
    representations for, use the FastText binary to run the model and
    extract the vectors from the output.

    Parameters
    ----------
    fasttext_binary_path: str
        Path to the FastText binary.

    fasttext_model_path: str
        Path to the trained FastText model on the same language that you want
        to generate representations of words from.

    input_words: List of str
        A list of strings to get FastText vectors for.

    verbose: boolean (optional, default=True)
        Log status messages to the console.

    Returns
    -------
    input_words_vectors_dict: dict
        A dictionary of string to numpy array of float, where the keys of the
        dictionary are the input words and the numpy array corresponding to
        each key is the associated vector as predicted by FastText.
    """
    # Write the input words to a temporary file (one on each line).
    input_words_tmp = tempfile.NamedTemporaryFile(delete=False)
    input_words_tempfile_path = input_words_tmp.name
    with open(input_words_tempfile_path, "w") as input_words_tmp_file:
        for input_word in input_words:
            input_words_tmp_file.write(input_word + "\n")

    # Use FastText to generate representations for the list of input words.
    cmd = [fasttext_binary_path, "print-word-vectors", fasttext_model_path]
    if verbose:
        logger.info("Running command: {} with input "
                    "file {}".format(cmd, input_words_tempfile_path))

    with open(input_words_tempfile_path) as input_file:
        raw_output = subprocess.check_output(
            cmd, stdin=input_file, universal_newlines=True)
    # Delete the temp file.
    os.unlink(input_words_tempfile_path)

    # Get the lines of output into a list.
    output_lines = raw_output.split("\n")
    output_lines.remove("")

    generated_vectors = {}

    for line in output_lines:
        split_line = line.rstrip().split(" ")
        word = split_line[0]
        vector = [float(i) for i in split_line[1:]]
        generated_vectors[word] = np.array(vector)

    assert set(generated_vectors.keys()) == set(input_words)

    return generated_vectors
