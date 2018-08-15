# Experiment Configs

This folder holds the various experiment configs used to train and run the OOV
translation solvers.

- [`similarity`](./similarity): These are the edit distance and vector distance
  based models.
  
- [`general_seq2seq_tests`](./general_seq2seq_tests): These configurations were
  used for verifying the seq2seq implementation with a simple copy task and WMT
  English to German neural machine translation.
  
- [`plural`](./plural): These configurations are used for running
  the [plural solvers](../oov/models/plural).

- [`seq2seq`](./seq2seq): These configurations are used for running
  the [seq2seq solvers](../oov/models/seq2seq), and are further categorized by
  what tokenization they use (e.g. character-level, morpheme-level, etc.).
