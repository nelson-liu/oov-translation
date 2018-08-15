import argparse
import gzip
import json
import logging
from lxml import etree as ET
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from oov.models import implemented_models

logger = logging.getLogger(__name__)
EMAIL_REGEX = re.compile(r"[^@]+@[^@]+\.[^@]+")
# From Django's validator
URL_REGEX = regex = re.compile(
    r"^(?:http|ftp)s?://"
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
    r"localhost|"
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"\[?[A-F0-9]*:[A-F0-9:]+\]?)"
    r"(?::\d+)?"
    r"(?:/?|[/?]\S+)$", re.IGNORECASE)


def main():
    argparser = argparse.ArgumentParser(
        description=("Post-edit ELISA pack with OOV translator."))
    argparser.add_argument("--elisa_path", type=str, required=True,
                           help=("The path to the ELISA language pack "
                                 "(a xml.gz file)."))
    argparser.add_argument("--oov_path", type=str,
                           help=("An optional path to a list of OOVs to "
                                 "replace in the ELISA output."))
    argparser.add_argument("--model_path", type=str, required=True,
                           help=("The path model to use to predict OOV "
                                 "translations."))
    argparser.add_argument("--predict_params", type=json.loads,
                           help=("Keyword arguments to pass to the model "
                                 "predict params script."))
    argparser.add_argument("--copy_vocab_path", type=str,
                           help=("Path to a list of words to copy if seen in the source "
                                 "text, one line each.."))
    argparser.add_argument("--output_dir", type=str, required=True,
                           help=("The directory to write post-edited "
                                 "translations to."))
    config = argparser.parse_args()
    original_sentences, oovs_to_translate = parse_elisa(
        config.elisa_path, config.oov_path)

    # Load the model from the path
    load_path = config.model_path
    load_folder = os.path.dirname(load_path)
    logger.info("Loading original config used to construct saved "
                "model from {}".format(load_folder))
    config_files = [os.path.join(load_folder, f) for f in
                    os.listdir(load_folder) if f.endswith("config.json")]
    if len(config_files) != 1:
        raise ValueError("Found config files {}. Please ensure that you "
                         "have exactly one config file in the same "
                         "directory as the model you wish to "
                         "load.".format(config_files))
    original_config_path = config_files[0]
    with open(original_config_path) as original_config_file:
        config_from_file = json.load(original_config_file)
    # Get the class referring to the selected model
    model_class_str = config_from_file.get("model_class")
    if model_class_str not in implemented_models:
        raise ValueError("{} is not a valid model class. "
                         "Possible choices are {}".format(
                             model_class_str, implemented_models))
    model_class = implemented_models[model_class_str]

    run_id = config_from_file.get("run_id")
    logger.info("Loading model checkpoint at {} "
                "for prediction".format(load_path))
    # Create an instance of the model
    model_params = config_from_file.get("model_params", {})
    model = model_class(**model_params)
    model.load_from_file(load_path)
    logger.info("Successfully loaded model!")

    predict_kwargs = config.predict_params if config.predict_params else {}
    translated_sequences = model.translate_list(oovs_to_translate, **predict_kwargs)
    assert len(translated_sequences) == len(oovs_to_translate)
    # Make a map from oovs_to_translate to translated_sequences
    oov_translations = {oov: translation for oov, translation in
                        zip(oovs_to_translate, translated_sequences)}

    # Read the copy vocab, if provided
    logger.info("Loading copy vocab from {}".format(
        config.copy_vocab_path))
    copy_vocab = set()
    if config.copy_vocab_path is not None:
        with open(config.copy_vocab_path) as copy_vocab_file:
            for line in copy_vocab_file:
                copy_vocab.add(line.rstrip("\n"))
    logger.info("Loaded copy vocab from {}, {} items".format(
        config.copy_vocab_path, len(copy_vocab)))

    # Parse the elisa XML and post-edit/replace the TEXT field with the
    # postedited original sentences we predicted.
    with gzip.open(config.elisa_path) as elisa_xml:
        tree = ET.parse(elisa_xml)
        root = tree.getroot()
        assert len(tree.xpath("//SEGMENT")) == len(original_sentences)
        for sentence, xml in zip(original_sentences,
                                 tree.xpath("//SEGMENT")):
            sbmt_output_node = xml.find(".//TEXT")
            postedited_text = sbmt_output_node.text
            for word, to_translate in sentence:
                if to_translate:
                    # If the word is something to translate, get the translation.
                    predicted_translation = oov_translations.get(word, word)
                    if (predicted_translation == "@@UNTRANSLATED_OOV@@" or
                            to_copy(word, copy_vocab)):
                        # We want to copy, so just don't replace.
                        continue
                    # Replace the first occurence of the word in our postedited text
                    # with the predicted translation.
                    postedited_text = postedited_text.replace(
                        word, predicted_translation, 1)
            # replace the SBMT output with our post-edited output
            sbmt_output_node.text = postedited_text

        elisa_filename = os.path.splitext(os.path.basename(config.elisa_path))[0]
        postedited_elisa_path = os.path.join(
            config.output_dir, elisa_filename + "." + run_id + ".postedited.xml.gz")
        logger.info("Done post-editing! Writing {}.".format(postedited_elisa_path))
        with gzip.open(postedited_elisa_path, 'wb') as postedited_elisa_gzip:
            ET.ElementTree(root).write(postedited_elisa_gzip, encoding="UTF-8",
                                       pretty_print=True)


def to_copy(input_oov, copy_vocab):
    """
    Given a word, decide whether to copy or translate it.

    Parameters
    ----------
    input_oov: str
        String representation of the input oov.

    copy_vocab: Set of str
        Set of str, if a string to translate occurs in the set,
        it is copied instead.

    Returns
    -------
    to_copy: boolean
        Whether or not to copy the translation.
    """
    if len(input_oov) == 0:
        return False
    if input_oov in copy_vocab:
        return True
    # If the first letter is capitalized
    if input_oov.isupper():
        return True
    # If it is an email
    if EMAIL_REGEX.match(input_oov):
        return True
    # If it is a twitter hashtag or mention
    if input_oov[0] == "@" or input_oov[0] == "#":
        return True
    if input_oov.lower() == "rt":
        return True
    # If it is a URL
    if URL_REGEX.match(input_oov):
        return True
    return False


def parse_elisa(elisa_path, oov_path=None):
    """
    Parse the ELISA package, extracting the original strings and the
    OOV strings.

    Parameters
    ----------
    elisa_path: str
        Path to the ELISA output, a xml.gz file.

    oov_path: str, optional (default=None)
        Path to a text file of OOVs to translate.

    Returns
    -------
    original_sentences: List of List[(str, bool)]
        The original sentences as generated by ELISA. Each tuple
        is a tuple of (original_string, to_translate), where
        to_translate indicates whether we should translate the word
        with the OOV system.

    oovs_to_translate: List[str]
        A list of OOVs to get translations for.
    """
    original_sentences = []
    oovs_to_translate = set()
    with gzip.open(elisa_path) as elisa_xml:
        tree = ET.parse(elisa_xml)
        for seg_el in tree.xpath("//SEGMENT"):
            tok_els = seg_el.findall(".//TOKENIZED_TARGET/TOKEN")

            tokens = [tok.text for tok in tok_els]
            tags = [tok.attrib["rule-class"] for tok in tok_els]

            original_sentence = []
            for idx, token in enumerate(tokens):
                is_oov = "unk" in tags[idx].lower()
                original_sentence.append((token, is_oov))
                # We get the OOVs from the oov_path if it is provided
                if is_oov and oov_path is None:
                    oovs_to_translate.add(token)
            original_sentences.append(original_sentence)
    if oov_path:
        with open(oov_path) as oov_file:
            for line in oov_file:
                oov = line.rstrip("\n")
                oovs_to_translate.add(oov)

    oovs_to_translate = list(oovs_to_translate)
    return (original_sentences, oovs_to_translate)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()
