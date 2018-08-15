from __future__ import division
import argparse
import logging
import mmap

import spacy
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main():
    argparser = argparse.ArgumentParser(
        description=("Given a TSV of <english><tab><german>, use SpaCy to tokenize."))
    argparser.add_argument("--input_tsv", type=str, required=True,
                           help=("The path to the data file to tokenize."))
    argparser.add_argument("--max_src_len", type=int, default=50,
                           help=("The maximum length of a source sentence."))
    argparser.add_argument("--max_tgt_len", type=int, default=50,
                           help=("The maximum length of a target sentence."))

    config = argparser.parse_args()

    logger.info("Initializing SpaCy English and German models.")
    en_nlp = spacy.load('en')
    de_nlp = spacy.load('de')
    output_lines = []

    skipped = 0
    with open(config.input_tsv) as input_tsv:
        for line in tqdm(input_tsv, total=get_line_number(config.input_tsv)):
            try:
                english, german = line.split("\t")
            except:
                skipped += 1
                continue
            # Tokenize English and German
            tokenized_english = [tok.text for tok in
                                 en_nlp.tokenizer(english.strip())]
            if (len(tokenized_english) > config.max_src_len or
                    len(tokenized_english) < 1):
                skipped += 1
                continue
            tokenized_german = [tok.text for tok in
                                de_nlp.tokenizer(german.strip())]
            if (len(tokenized_german) > config.max_tgt_len or
                    len(tokenized_german) < 1):
                skipped += 1
                continue
            english_tok = ' '.join(tokenized_english)
            german_tok = ' '.join(tokenized_german)
            output_lines.append(english_tok + "\t" + german_tok)

    logger.info("Skipped {} instances".format(skipped))
    # Write all the output lines to file
    with open(config.input_tsv + ".tokenized", "w") as output_tsv:
        for line in tqdm(output_lines):
            output_tsv.write("{}\n".format(line))


def get_line_number(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()
