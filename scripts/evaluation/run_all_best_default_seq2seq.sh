set -e

# Amharic run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_amh_weighted_run0/seq2seq_char_amh_weighted_run0-step-570000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/amh-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_amh_weighted_run0/seq2seq_char_amh_weighted_run0-step-570000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/amh-eng/test
# Amharic run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_amh_weighted_run1/seq2seq_char_amh_weighted_run1-step-480000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/amh-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_amh_weighted_run1/seq2seq_char_amh_weighted_run1-step-480000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/amh-eng/test
# Amharic run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_amh_weighted_run2/seq2seq_char_amh_weighted_run2-step-370000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/amh-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_amh_weighted_run2/seq2seq_char_amh_weighted_run2-step-370000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/amh-eng/test

# Arabic run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ara_weighted_run0/seq2seq_char_ara_weighted_run0-step-850000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ara-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ara_weighted_run0/seq2seq_char_ara_weighted_run0-step-850000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ara-eng/test
# Arabic run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ara_weighted_run1/seq2seq_char_ara_weighted_run1-step-860000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ara-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ara_weighted_run1/seq2seq_char_ara_weighted_run1-step-860000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ara-eng/test
# Arabic run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ara_weighted_run2/seq2seq_char_ara_weighted_run2-step-870000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ara-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ara_weighted_run2/seq2seq_char_ara_weighted_run2-step-870000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ara-eng/test

# Bengali run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ben_weighted_run0/seq2seq_char_ben_weighted_run0-epoch-18.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ben-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ben_weighted_run0/seq2seq_char_ben_weighted_run0-epoch-18.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ben-eng/test
# Bengali run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ben_weighted_run1/seq2seq_char_ben_weighted_run1-step-990000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ben-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ben_weighted_run1/seq2seq_char_ben_weighted_run1-step-990000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ben-eng/test
# Bengali run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ben_weighted_run2/seq2seq_char_ben_weighted_run2-step-1020000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ben-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_ben_weighted_run2/seq2seq_char_ben_weighted_run2-step-1020000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/ben-eng/test

# Chinese Mandarin run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_cmn_weighted_run0/seq2seq_char_cmn_weighted_run0-epoch-15.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/cmn-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_cmn_weighted_run0/seq2seq_char_cmn_weighted_run0-epoch-15.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/cmn-eng/test
# Chinese Mandarin run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_cmn_weighted_run1/seq2seq_char_cmn_weighted_run1-step-490000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/cmn-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_cmn_weighted_run1/seq2seq_char_cmn_weighted_run1-step-490000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/cmn-eng/test
# Chinese Mandarin run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_cmn_weighted_run2/seq2seq_char_cmn_weighted_run2-step-150000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/cmn-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_cmn_weighted_run2/seq2seq_char_cmn_weighted_run2-step-150000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/cmn-eng/test

# Farsi run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_fas_weighted_run0/seq2seq_char_fas_weighted_run0-step-300000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/fas-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_fas_weighted_run0/seq2seq_char_fas_weighted_run0-step-300000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/fas-eng/test
# Farsi run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_fas_weighted_run1/seq2seq_char_fas_weighted_run1-step-420000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/fas-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_fas_weighted_run1/seq2seq_char_fas_weighted_run1-step-420000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/fas-eng/test
# Farsi run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_fas_weighted_run2/seq2seq_char_fas_weighted_run2-epoch-18.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/fas-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_fas_weighted_run2/seq2seq_char_fas_weighted_run2-epoch-18.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/fas-eng/test

# Hausa run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hau_weighted_run0/seq2seq_char_hau_weighted_run0-step-400000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hau-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hau_weighted_run0/seq2seq_char_hau_weighted_run0-step-400000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hau-eng/test
# Hausa run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hau_weighted_run1/seq2seq_char_hau_weighted_run1-step-440000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hau-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hau_weighted_run1/seq2seq_char_hau_weighted_run1-step-440000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hau-eng/test
# Hausa run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hau_weighted_run2/seq2seq_char_hau_weighted_run2-step-370000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hau-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hau_weighted_run2/seq2seq_char_hau_weighted_run2-step-370000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hau-eng/test

# Hungarian run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hun_weighted_run0/seq2seq_char_hun_weighted_run0-step-1310000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hun-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hun_weighted_run0/seq2seq_char_hun_weighted_run0-step-1310000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hun-eng/test
# Hungarian run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hun_weighted_run1/seq2seq_char_hun_weighted_run1-step-1370000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hun-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hun_weighted_run1/seq2seq_char_hun_weighted_run1-step-1370000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hun-eng/test
# Hungarian run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hun_weighted_run2/seq2seq_char_hun_weighted_run2-step-1230000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hun-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_hun_weighted_run2/seq2seq_char_hun_weighted_run2-step-1230000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/hun-eng/test

# Russian run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_rus_weighted_run0/seq2seq_char_rus_weighted_run0-step-770000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/rus-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_rus_weighted_run0/seq2seq_char_rus_weighted_run0-step-770000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/rus-eng/test
# Russian run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_rus_weighted_run1/seq2seq_char_rus_weighted_run1-step-910000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/rus-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_rus_weighted_run1/seq2seq_char_rus_weighted_run1-step-910000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/rus-eng/test
# Russian run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_rus_weighted_run2/seq2seq_char_rus_weighted_run2-step-890000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/rus-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_rus_weighted_run2/seq2seq_char_rus_weighted_run2-step-890000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/rus-eng/test

# Somali run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_som_weighted_run0/seq2seq_char_som_weighted_run0-step-360000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/som-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_som_weighted_run0/seq2seq_char_som_weighted_run0-step-360000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/som-eng/test
# Somali run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_som_weighted_run1/seq2seq_char_som_weighted_run1-epoch-24.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/som-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_som_weighted_run1/seq2seq_char_som_weighted_run1-epoch-24.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/som-eng/test
# Somali run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_som_weighted_run2/seq2seq_char_som_weighted_run2-epoch-30.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/som-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_som_weighted_run2/seq2seq_char_som_weighted_run2-epoch-30.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/som-eng/test

# Spanish run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_spa_weighted_run0/seq2seq_char_spa_weighted_run0-step-890000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/spa-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_spa_weighted_run0/seq2seq_char_spa_weighted_run0-step-890000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/spa-eng/test
# Spanish run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_spa_weighted_run1/seq2seq_char_spa_weighted_run1-step-1250000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/spa-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_spa_weighted_run1/seq2seq_char_spa_weighted_run1-step-1250000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/spa-eng/test
# Spanish run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_spa_weighted_run2/seq2seq_char_spa_weighted_run2-step-1350000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/spa-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_spa_weighted_run2/seq2seq_char_spa_weighted_run2-step-1350000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/spa-eng/test

# Tamil run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tam_weighted_run0/seq2seq_char_tam_weighted_run0-epoch-42.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tam-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tam_weighted_run0/seq2seq_char_tam_weighted_run0-epoch-42.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tam-eng/test
# Tamil run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tam_weighted_run1/seq2seq_char_tam_weighted_run1-epoch-42.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tam-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tam_weighted_run1/seq2seq_char_tam_weighted_run1-epoch-42.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tam-eng/test
# Tamil run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tam_weighted_run2/seq2seq_char_tam_weighted_run2-step-360000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tam-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tam_weighted_run2/seq2seq_char_tam_weighted_run2-step-360000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tam-eng/test

# Turkish run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tur_weighted_run0/seq2seq_char_tur_weighted_run0-step-660000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tur-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tur_weighted_run0/seq2seq_char_tur_weighted_run0-step-660000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tur-eng/test
# Turkish run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tur_weighted_run1/seq2seq_char_tur_weighted_run1-step-750000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tur-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tur_weighted_run1/seq2seq_char_tur_weighted_run1-step-750000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tur-eng/test
# Turkish run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tur_weighted_run2/seq2seq_char_tur_weighted_run2-step-730000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tur-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_tur_weighted_run2/seq2seq_char_tur_weighted_run2-step-730000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/tur-eng/test

# Urdu run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_urd_weighted_run0/seq2seq_char_urd_weighted_run0-step-540000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/urd-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_urd_weighted_run0/seq2seq_char_urd_weighted_run0-step-540000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/urd-eng/test
# Urdu run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_urd_weighted_run1/seq2seq_char_urd_weighted_run1-step-550000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/urd-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_urd_weighted_run1/seq2seq_char_urd_weighted_run1-step-550000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/urd-eng/test
# Urdu run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_urd_weighted_run2/seq2seq_char_urd_weighted_run2-step-670000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/urd-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_urd_weighted_run2/seq2seq_char_urd_weighted_run2-step-670000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/urd-eng/test

# Uzbek run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_uzb_weighted_run0/seq2seq_char_uzb_weighted_run0-step-440000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/uzb-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_uzb_weighted_run0/seq2seq_char_uzb_weighted_run0-step-440000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/uzb-eng/test
# Uzbek run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_uzb_weighted_run1/seq2seq_char_uzb_weighted_run1-step-480000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/uzb-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_uzb_weighted_run1/seq2seq_char_uzb_weighted_run1-step-480000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/uzb-eng/test
# Uzbek run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_uzb_weighted_run2/seq2seq_char_uzb_weighted_run2-step-460000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/uzb-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_uzb_weighted_run2/seq2seq_char_uzb_weighted_run2-step-460000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/uzb-eng/test

# Vietnamese run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_vie_weighted_run0/seq2seq_char_vie_weighted_run0-step-70000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/vie-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_vie_weighted_run0/seq2seq_char_vie_weighted_run0-step-70000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/vie-eng/test
# Vietnamese run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_vie_weighted_run1/seq2seq_char_vie_weighted_run1-step-620000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/vie-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_vie_weighted_run1/seq2seq_char_vie_weighted_run1-step-620000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/vie-eng/test
# Vietnamese run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_vie_weighted_run2/seq2seq_char_vie_weighted_run2-step-240000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/vie-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_vie_weighted_run2/seq2seq_char_vie_weighted_run2-step-240000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/vie-eng/test

# Yoruba run 0
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_yor_weighted_run0/seq2seq_char_yor_weighted_run0-epoch-20.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/yor-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_yor_weighted_run0/seq2seq_char_yor_weighted_run0-epoch-20.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/yor-eng/test
# Yoruba run 1
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_yor_weighted_run1/seq2seq_char_yor_weighted_run1-step-580000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/yor-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_yor_weighted_run1/seq2seq_char_yor_weighted_run1-step-580000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/yor-eng/test
# Yoruba run 2
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_yor_weighted_run2/seq2seq_char_yor_weighted_run2-step-530000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/yor-eng/dev
python scripts/run/run_oov_solver.py \
       --predict_from_path=./models/seq2seq/char/seq2seq_char_yor_weighted_run2/seq2seq_char_yor_weighted_run2-step-530000.ckpt \
       --eval_path=./data/processed/language_pairs_v3.2_gold_aligned/yor-eng/test
