import argparse
from collections import Counter
import logging
import os

from tqdm import tqdm

logger = logging.getLogger(__name__)


def main():
    argparser = argparse.ArgumentParser(
        description=("Given language pair data, output a datafile with all "
                     "the english words in the lexicon and training data."))
    argparser.add_argument("--data_dir", type=str, required=True,
                           help=("The path to the directory with the "
                                 "language pair data folders (e.g. hau-eng, uzb-eng). "
                                 "For each language, we will pull from the lexicon and "
                                 "train.phrases data files."))
    argparser.add_argument("--output_dir", type=str, required=True,
                           help=("The path to write the generated "
                                 "data files for each language."))

    config = argparser.parse_args()
    language_pair_dirs = sorted(
        [os.path.join(config.data_dir, name) for name in
         os.listdir(config.data_dir) if
         os.path.isdir(os.path.join(config.data_dir, name))])

    english_words = Counter()
    for language_pair_dir in tqdm(language_pair_dirs):

        language_pair_str = os.path.basename(language_pair_dir)

        train_file_path = os.path.join(language_pair_dir, "train")
        if not os.path.exists(train_file_path):
            logger.warning("Train for {} does not exist, "
                           "skipping.".format(language_pair_str))
            continue
        with open(train_file_path) as train_file:
            for line in train_file:
                source_sentence, target_sentence = line.rstrip("\n").split("\t")
                english_words.update(list(map(str.strip, target_sentence.split(" "))))

        lexicon_file_path = os.path.join(language_pair_dir, "lexicon")
        if not os.path.exists(lexicon_file_path):
            logger.warning("Lexicon for {} does not exist, "
                           "skipping.".format(language_pair_str))
            continue
        with open(lexicon_file_path) as lexicon_file:
            for line in lexicon_file:
                source_word, pos, target_word = line.rstrip("\n").split("\t")
                english_words.update(list(map(str.strip, target_word.split(" "))))

        tgt_given_src_file_path = os.path.join(language_pair_dir, "tgt_given_src")
        if not os.path.exists(tgt_given_src_file_path):
            logger.warning("tgt_given_src for {} does not exist, "
                           "skipping.".format(language_pair_str))
            continue
        with open(tgt_given_src_file_path) as tgt_given_src_file:
            for line in tgt_given_src_file:
                target_word, source_word, frequency = line.split(" ")
                if any(char.isalpha() for char in target_word):
                    english_words.update(list(map(str.strip, target_word.split(" "))))

    # Take the set and turn it into a list of tuples, removing the whitespace characters
    english_words_list = []
    for word, count in english_words.items():
        if word != "" and not word.isspace():
            english_words_list.append((word, count))

    # Sort the tuples for reproducibility
    sorted_english_words_list = sorted(english_words_list, key=lambda x: x[1],
                                       reverse=True)

    output_path = os.path.join(config.output_dir, "dataset_english_types.txt".format())
    with open(output_path, 'w') as output_file:
        for word, count in sorted_english_words_list:
            output_file.write("{}\t{}\n".format(word, count))
    logger.info("Wrote {} words and counts to {}".format(len(english_words_list),
                                                         output_path))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()
