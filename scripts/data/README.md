# Data Scripts

This directory contains the scripts that are used to create model-specific
datasets from the original OOV dataset (`language_pairs_v3.2`).

This README seeks to document what the function of each script is, what format
of data it takes and produces, and what models use it.

- `create_plural_translation_data.py` / `create_all_plural_translation_data.sh`
  - `create_plural_translation_data.py`, as its name suggests, generates plural
    translation data for a single language. 

    - The output format is `<english singular>\t<foreign singular>\t<english
      plural>\t<foreign_plural>`. 

    - It takes as input a `tgt_given_src` t-table and a `src_given_tgt` t-table
      from the original OOV dataset, the path to a custom morphology cache
      provided by Ulf Hermjakob, the path to the `uroman` executable, and lastly
      the language code to pass to `uroman`. This data is used to train
      the [`PluralStringSolver`](./../../oov/models/plural/plural_string_solver).
    
  - `create_all_plural_translation_data.sh` is a convenient wrapper script for quickly
    generating plural translation data for each of the 16 languages in the dataset. It
    simply runs `create_plural_translation_data.py` 16 times.
    
- `create_word_translation_data.py`
  - This script creates data where the task is to translate one word to another
    word. 
    
  - The output format is `<foreign word>\t<english word>`. 
  
  - It takes as input `tgt_given_src` t-table, a `lexicon` (bilingual
    dictionary), and `train.phrases` (phrase table / phrases that align to each
    other).
    
- `elisa2flat.py`
  - This script was written by Jon May, and is only compatible with Python 3. It
    takes a raw ELISA XML corpus as input, a list of fields, and outputs a text
    file monolingual corpus (useful for, e.g., learning word vectors).

- `get_english_types.py`
  - This script takes all the English words in a dataset from the original OOV
    data. Used to send Penn the various English words we wanted morphological
    segmentations of.

- `tokenize_en_de.py`
  - This script was used when testing EN to DE translation --- it tokenized the
    text to return `<english sentence>\t<german sentence>`, where each sentence
    is a list of space-delimited tokens.
