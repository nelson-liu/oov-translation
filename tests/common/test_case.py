from unittest import TestCase
import logging
import hashlib
import os
import shutil
import subprocess
import tempfile
import zipfile

logger = logging.getLogger(__name__)


class OOVTestCase(TestCase):
    # Directory where everything temporary and test-related is written
    test_dir = tempfile.mkdtemp()

    # Path to the uroman executable
    uroman_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               os.pardir, os.pardir, 'util', "uroman-v1.2",
                               "bin", "uroman.pl")

    # Folder for Vanilla OOV dataset samples (in Amharic), and
    # logs/saved models using them
    language_pair_data_dir = os.path.join(test_dir, "language_pairs", "amh-eng")

    # Folder for OOV translation dataset samples (e.g. for training
    # seq2seq char models), and logs/saved models using them.
    language_pair_word_translation_data_dir = os.path.join(
        test_dir, "language_pairs_word_translation", "spa-eng")

    # Paths for (normal) word-level translation tests
    language_pair_translation_data_dir = os.path.join(
        test_dir, "language_pair_data_translation", "spa-eng")

    # Paths for plural solver
    plural_solver_data_dir = os.path.join(
        test_dir, "plural_translation_data")

    def setUp(self):
        logging.basicConfig(format=('%(asctime)s - %(levelname)s - '
                                    '%(name)s - %(message)s'),
                            level=logging.INFO)

        # Create test dir for the vanilla OOV dataset
        if not os.path.exists(self.language_pair_data_dir):
            os.makedirs(self.language_pair_data_dir)

        # Create test dir for the OOV word translation dataset
        if not os.path.exists(self.language_pair_word_translation_data_dir):
            os.makedirs(self.language_pair_word_translation_data_dir)

        # Create test dir for normal NMT dataset
        if not os.path.exists(self.language_pair_translation_data_dir):
            os.makedirs(self.language_pair_translation_data_dir)

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except:
            subprocess.call(["rm", "-rf", self.test_dir])

    def download_amh_fasttext(self):
        archive_paths = ["{}/oov_data/wiki.am.zip".format(os.environ["HOME"]),
                         "/stage/oov_data/wiki.am.zip"]
        correct_archive_md5_sum = "fe2c58a3930f727c0c2244c56bbc371a"
        correct_vectors_md5_sum = "dc9160687920364d8c59bdc3e828a1ce"
        correct_model_md5_sum = "fd914f2e59d685c827ee91dae1719f3e"
        if not os.path.exists(os.path.join(self.language_pair_data_dir,
                                           "amh")):
            os.makedirs(os.path.join(self.language_pair_data_dir,
                                     "amh"))

        to_unzip = True
        to_download = True
        # Check if an unzipped version already exists in this folder, and if it has
        # the right md5 if so, copy the unzipped files over and set to_unzip to False
        if (os.path.exists("{}/oov_data/wiki.am.bin".format(os.environ["HOME"])) and
                (self.md5("{}/oov_data/wiki.am.bin".format(os.environ["HOME"])) ==
                 correct_model_md5_sum) and
                os.path.exists("{}/oov_data/wiki.am.vec".format(os.environ["HOME"])) and
                (self.md5("{}/oov_data/wiki.am.vec".format(os.environ["HOME"])) ==
                 correct_vectors_md5_sum)):
            logger.info("Found unzipped files at {}, copying "
                        "them over.".format("{}/oov_data/".format(os.environ["HOME"])))
            shutil.copyfile("{}/oov_data/wiki.am.bin".format(os.environ["HOME"]),
                            os.path.abspath(
                                os.path.join(self.language_pair_data_dir, "amh",
                                             "wiki.am.bin")))
            shutil.copyfile("{}/oov_data/wiki.am.vec".format(os.environ["HOME"]),
                            os.path.abspath(
                                os.path.join(self.language_pair_data_dir, "amh",
                                             "wiki.am.vec")))
            to_unzip = False
            to_download = False
        elif (os.path.exists("/stage/oov_data/wiki.am.bin") and
                self.md5("/stage/oov_data/wiki.am.bin") == correct_model_md5_sum and
                os.path.exists("/stage/oov_data/wiki.am.vec") and
                self.md5("/stage/oov_data/wiki.am.vec") == correct_vectors_md5_sum):
            logger.info("Found unzipped files at {}, copying "
                        "them over.".format("/stage/oov_data/"))
            shutil.copyfile("/stage/oov_data/wiki.am.bin",
                            os.path.abspath(
                                os.path.join(self.language_pair_data_dir, "amh",
                                             "wiki.am.bin")))
            shutil.copyfile("/stage/oov_data/wiki.am.vec",
                            os.path.abspath(
                                os.path.join(self.language_pair_data_dir, "amh",
                                             "wiki.am.vec")))
            to_unzip = False
            to_download = False

        # Check the archive paths, and see if we need to download
        archive_path = archive_paths[0]
        for path in archive_paths:
            if (os.path.exists(path) and
                    self.md5(path) == correct_archive_md5_sum):
                to_download = False
                archive_path = path

        if to_download:
            url = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.am.zip"
            archive_path = os.path.join(self.language_pair_data_dir,
                                        "amh", "wiki.am.zip")
            args = ['wget', '-O', archive_path, url]
            logger.info("Downloading amh vectors "
                        "from {} to ~/oov_data".format(url))
            output = subprocess.Popen(args, stdout=subprocess.PIPE)
            out, err = output.communicate()

        # Unzip the archive we downloaded.
        if to_unzip:
            logger.info("Extracting downloaded zip archive at {}".format(archive_path))
            with zipfile.ZipFile(archive_path, "r") as zfile:
                zfile.extractall(os.path.join(self.language_pair_data_dir, "amh"))
        return {
            "fasttext_model_path": os.path.abspath(
                os.path.join(self.language_pair_data_dir, "amh",
                             "wiki.am.bin")),
            "fasttext_vectors_path": os.path.abspath(
                os.path.join(self.language_pair_data_dir, "amh",
                             "wiki.am.vec"))
        }

    def md5(self, fname):
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def write_language_pair_data(self):
        # Make the directories for logs and models
        self.language_pair_log_dir = os.path.join(self.language_pair_data_dir, "logs")
        self.language_pair_save_dir = os.path.join(self.language_pair_data_dir, "models")
        os.makedirs(self.language_pair_log_dir)
        os.makedirs(self.language_pair_save_dir)

        with open(os.path.join(self.language_pair_data_dir,
                               "dev"), 'w') as amh_dev:
            amh_dev.write("amhword1\tamhword1translation\treal amhword1 context"
                          "\tsome amhword1translation ctext\t1-1\n")
            amh_dev.write("amhword2\tamhword2translation\ta small amhword2 window"
                          "\trandom amhword2translation background\t2-1\n")

        with open(os.path.join(self.language_pair_data_dir,
                               "lexicon"), 'w') as amh_lexicon:
            amh_lexicon.write("amhword1notoov\tNOUN\tamhword1 not oov translation\n")
            amh_lexicon.write("amhword2notoov\tNOUN\tamhword2notoov translation\n")
            amh_lexicon.write("amhword3notoov\tNOUN\tamhword3notoovtranslation\n")

        with open(os.path.join(self.language_pair_data_dir,
                               "src_given_tgt"), 'w') as amh_src_given_tgt:
            amh_src_given_tgt.write("somesource1 sometarget1 0.5\n")
            amh_src_given_tgt.write("somesource2 sometarget1 0.5\n")
            amh_src_given_tgt.write("somesource3 sometarget2 1.0\n")
            amh_src_given_tgt.write("amhword1close amhword1translation 0.75\n")
            amh_src_given_tgt.write("amhword1further amhword1translation 0.25\n")
            amh_src_given_tgt.write("amhword2close amhword2translation 0.2\n")
            amh_src_given_tgt.write("amhword2clsr amhword2translation 0.7\n")
            amh_src_given_tgt.write("amhword2veryfurther amhword2translation 0.1\n")

        with open(os.path.join(self.language_pair_data_dir,
                               "src_given_tgt.unnormalized"),
                  'w') as amh_src_given_tgt_unnorm:
            amh_src_given_tgt_unnorm.write("somesource1 sometarget1 4\n")
            amh_src_given_tgt_unnorm.write("somesource2 sometarget1 4\n")
            amh_src_given_tgt_unnorm.write("somesource3 sometarget2 1\n")
            amh_src_given_tgt_unnorm.write("amhword1close amhword1translation 12\n")
            amh_src_given_tgt_unnorm.write("amhword1further amhword1translation 4\n")
            amh_src_given_tgt_unnorm.write("amhword2close amhword2translation 2\n")
            amh_src_given_tgt_unnorm.write("amhword2clsr amhword2translation 7\n")
            amh_src_given_tgt_unnorm.write("amhword2veryfurther amhword2translation 1\n")

        with open(os.path.join(self.language_pair_data_dir,
                               "tgt_given_src"), 'w') as amh_tgt_given_src:
            amh_tgt_given_src.write("sometarget1 somesource1 0.7\n")
            amh_tgt_given_src.write("sometarget2 somesource1 0.3\n")
            amh_tgt_given_src.write("sometarget2 somesource3 1.0\n")
            amh_tgt_given_src.write("amhword1translation1-1 amhword1close 0.8\n")
            amh_tgt_given_src.write("amhword1translation1-2 amhword1close 0.1\n")
            amh_tgt_given_src.write("amhword1translation1-3 amhword1close 0.1\n")
            amh_tgt_given_src.write("amhword1translation-1 amhword1further 0.4\n")
            amh_tgt_given_src.write("amhword1translation-2 amhword1further 0.6\n")
            amh_tgt_given_src.write("amhword2translation-1 amhword2close 1.0\n")
            amh_tgt_given_src.write("amhword2translation-1 amhword2clsr 0.45\n")
            amh_tgt_given_src.write("amhword2translation-2 amhword2clsr 0.55\n")
            amh_tgt_given_src.write("amhword2translation-3 amhword2veryfurther 1.0\n")

        with open(os.path.join(self.language_pair_data_dir,
                               "tgt_given_src.unnormalized"),
                  'w') as amh_tgt_given_src_unnorm:
            amh_tgt_given_src_unnorm.write("sometarget1 somesource1 7\n")
            amh_tgt_given_src_unnorm.write("sometarget2 somesource1 3\n")
            amh_tgt_given_src_unnorm.write("sometarget2 somesource3 1\n")
            amh_tgt_given_src_unnorm.write("amhword1translation1-1 amhword1close 80\n")
            amh_tgt_given_src_unnorm.write("amhword1translation1-2 amhword1close 10\n")
            amh_tgt_given_src_unnorm.write("amhword1translation1-3 amhword1close 10\n")
            amh_tgt_given_src_unnorm.write("amhword1translation-1 amhword1further 4\n")
            amh_tgt_given_src_unnorm.write("amhword1translation-2 amhword1further 6\n")
            amh_tgt_given_src_unnorm.write("amhword2translation-1 amhword2close 10\n")
            amh_tgt_given_src_unnorm.write("amhword2translation-1 amhword2clsr 45\n")
            amh_tgt_given_src_unnorm.write("amhword2translation-2 amhword2clsr 55\n")
            amh_tgt_given_src_unnorm.write("amhword2translation-3 "
                                           "amhword2veryfurther 10\n")

    def write_word_translation_data(self):
        # Make folders for the logs and models
        self.language_pair_word_translation_log_dir = os.path.join(
            self.language_pair_word_translation_data_dir, "logs")
        self.language_pair_word_translation_save_dir = os.path.join(
            self.language_pair_word_translation_data_dir, "models")
        os.makedirs(self.language_pair_word_translation_log_dir)
        os.makedirs(self.language_pair_word_translation_save_dir)

        # Make paths for the data files
        self.spa_word_translation_train = os.path.join(
            self.language_pair_word_translation_data_dir, "spa-eng.train")
        self.spa_word_translation_val = os.path.join(
            self.language_pair_word_translation_data_dir, "spa-eng.val")
        self.spa_word_translation_test = os.path.join(
            self.language_pair_word_translation_data_dir, "spa-eng.test")
        self.hun_word_translation_train = os.path.join(
            self.language_pair_word_translation_data_dir, "hun-eng.train")
        self.hun_word_translation_val = os.path.join(
            self.language_pair_word_translation_data_dir, "hun-eng.val")
        self.hun_word_translation_test = os.path.join(
            self.language_pair_word_translation_data_dir, "hun-eng.test")

        self.spa_word_translation_src_segs = os.path.join(
            self.language_pair_word_translation_data_dir, "unittest_es.seg")
        self.spa_word_translation_tgt_segs = os.path.join(
            self.language_pair_word_translation_data_dir, "unittest_en.seg")

        with open(self.spa_word_translation_train, 'w') as spa_train:
            spa_train.write("perro\tdog\n")
            spa_train.write("perros\tdogs\n")
            spa_train.write("perrito\tlittle dog\n")
            spa_train.write("gato\tcat\n")
            spa_train.write("amarillo\tyellow\n")
            spa_train.write("amarilla\tyellow\n")
            spa_train.write("amarillos\tyellow\n")
            spa_train.write("gatos\tcats\n")
        with open(self.spa_word_translation_val, 'w') as spa_val:
            spa_val.write("perrita\tlittle dog\n")
            spa_val.write("gatito\tlittle cat\n")
        with open(self.spa_word_translation_test, 'w') as spa_test:
            spa_test.write("perritos\tlittle dogs\n")
            spa_test.write("gatita\tlittle cat\n")
            spa_test.write("gattitas\tlittle cats\n")

        with open(self.hun_word_translation_train, 'w') as hun_train:
            hun_train.write("kutya\tdog\n")
            hun_train.write("kutyak\tdogs\n")
            hun_train.write("kis kutya\tlittle dog\n")
            hun_train.write("cica\tcat\n")
            hun_train.write("sarga\tyellow\n")
            hun_train.write("sargit\tyellow\n")
            hun_train.write("macskak\tcats\n")
        with open(self.hun_word_translation_val, 'w') as hun_val:
            hun_val.write("kolyokkutya\tlittle dog\n")
            hun_val.write("kismacska\tlittle cat\n")
        with open(self.hun_word_translation_test, 'w') as hun_test:
            hun_test.write("kolykok\tlittle dogs\n")
            hun_test.write("cica\tlittle cat\n")
            hun_test.write("kiscicak\tlittle cats\n")

        with open(self.spa_word_translation_src_segs, "w") as src_segs:
            src_segs.write("perrito\tperrito\tperrito $ $\n")
            src_segs.write("perritos\tperrito s\tperrito $ $ perrito $ s\n")
            src_segs.write("gato\tgato\tgato $ $\n")
            src_segs.write("amarillo\tamarill o\tamarill $ $ amarill $ o\n")
            src_segs.write("amarillas\tamarill a s\t"
                           "amarill $ $ amarill $ a amarilla $ s\n")
            src_segs.write("perritos\tperrito s\tperrito $ $ perrito $ s\n")

        with open(self.spa_word_translation_tgt_segs, "w") as tgt_segs:
            tgt_segs.write("dog\tdog\tdog $ $\n")
            tgt_segs.write("dogs\tdog s\tdog $ $ dog $ s\n")
            tgt_segs.write("little\tlitt le\tlitte $ $ litte DEL-e le\n")
            tgt_segs.write("cats\tcat s\tcat $ $ cat $ s\n")
            tgt_segs.write("yellow\tyellow\tyellow $ $\n")

    def write_translation_data(self):
        self.language_pair_translation_log_dir = os.path.join(
            self.language_pair_translation_data_dir, "logs")
        self.language_pair_translation_save_dir = os.path.join(
            self.language_pair_translation_data_dir, "models")
        os.makedirs(self.language_pair_translation_log_dir)
        os.makedirs(self.language_pair_translation_save_dir)
        self.language_pair_translation_train = os.path.join(
            self.language_pair_translation_data_dir, "spa-eng.train")
        self.language_pair_translation_val = os.path.join(
            self.language_pair_translation_data_dir, "spa-eng.val")
        self.language_pair_translation_test = os.path.join(
            self.language_pair_translation_data_dir, "spa-eng.test")

        with open(self.language_pair_translation_train, 'w') as spa_train:
            spa_train.write("Me gustan los perros\tI like dogs\n")
            spa_train.write("Me gustan los gatos\tI like cats\n")
            spa_train.write("El gatito es muy pequeno\tThe little cat is very small\n")
            spa_train.write("La gatita es muy pequena\tThe little cat is very small\n")
            spa_train.write("el perro tiene hambre\tthe dog is hungry\n")
            spa_train.write("la perra tiene hambre\tthe dog is hungry\n")
        with open(self.language_pair_translation_val, 'w') as spa_val:
            spa_val.write("el perrito es muy pequeno\tthe little cat is very small\n")
            spa_val.write("la perrita es pequena\tthe little dog is small\n")
            spa_val.write("La gatita tiene hambra\tThe little cat is hungry\n")
        with open(self.language_pair_translation_test, 'w') as spa_test:
            spa_test.write("la perrita tiene hambre\tthe little dog is hungry\n")
            spa_test.write("el perrito tiene hambre\tthe little dog is hungry\n")
            spa_test.write("los gatos tienen hambre\tthe cats are hungry\n")

    def write_plural_solver_data(self):
        # Make the directories for logs and models
        self.plural_solver_log_dir = os.path.join(
            self.plural_solver_data_dir, "logs")
        self.plural_solver_save_dir = os.path.join(
            self.plural_solver_data_dir, "models")
        os.makedirs(self.plural_solver_log_dir)
        os.makedirs(self.plural_solver_save_dir)

        self.plural_solver_en_fr_sing_plural_pairs_train = os.path.join(
            self.plural_solver_data_dir,
            "plural_solver_en_fr_sing_plural_pairs_train.txt")
        self.plural_solver_en_fr_sing_plural_pairs_val = os.path.join(
            self.plural_solver_data_dir,
            "plural_solver_en_fr_sing_plural_pairs_val.txt")
        self.plural_solver_tgt_given_src_path = os.path.join(
            self.plural_solver_data_dir, "plural_solver_tgt_given_src")
        self.plural_solver_src_given_tgt_path = os.path.join(
            self.plural_solver_data_dir, "plural_solver_src_given_tgt")
        self.plural_solver_vectors_path = os.path.join(
            self.plural_solver_data_dir, "plural_solver_vectors.txt")

        with open(self.plural_solver_tgt_given_src_path, 'w') as tgt_given_src_file:
            tgt_given_src_file.write("cats gatos 10\n")
            tgt_given_src_file.write("cats gatoss 3\n")
            tgt_given_src_file.write("cats gatas 7\n")
            tgt_given_src_file.write("cat gato 9\n")
            tgt_given_src_file.write("dog perro 11\n")
            tgt_given_src_file.write("dogs perros 12\n")
            tgt_given_src_file.write("snakes serpientes 12\n")
            tgt_given_src_file.write("snake serpiente 13\n")

        with open(self.plural_solver_src_given_tgt_path, "w") as src_given_tgt_file:
            src_given_tgt_file.write("gatos cats 7\n")
            src_given_tgt_file.write("gatitos cats 4\n")
            src_given_tgt_file.write("gato cat 9\n")
            src_given_tgt_file.write("perro dog 11\n")
            src_given_tgt_file.write("doggo dog 3\n")
            src_given_tgt_file.write("perros dogs 12\n")
            src_given_tgt_file.write("serpientes snakes 12\n")
            src_given_tgt_file.write("serpiente snake 13\n")

        with open(self.plural_solver_en_fr_sing_plural_pairs_train, "w") as en_fr_pairs:
            en_fr_pairs.write("dog\tperro\tdogs\tperros\n")
            en_fr_pairs.write("cat\tgato\tcats\tgatos\n")
            en_fr_pairs.write("cat\tgata\tcats\tgatas\n")
            en_fr_pairs.write("snake\tserpiente\tsnakes\tserpientes\n")
        with open(self.plural_solver_en_fr_sing_plural_pairs_val, "w") as en_fr_pairs:
            en_fr_pairs.write("kitten\tgatito\tkittens\tgatitos\n")
            en_fr_pairs.write("party\tfiesta\tparties\tfiestas\n")

        with open(self.plural_solver_vectors_path, 'w') as vec_file:
            vec_file.write("6 5\n")
            vec_file.write("gato 0.1 0.8 0.1 0.9 0.2\n")
            vec_file.write("perro 0.1 0.3 0.8 0.2 0.3\n")
            vec_file.write("gatos 0.9 0.3 0.8 0.2 0.4\n")
            vec_file.write("perros 0.3 0.4 0.2 0.9 0.1\n")
            vec_file.write("amarillo 0.7 0.6 0.2 0.4 0.1\n")
            vec_file.write("amarillos 0.1 0.3 0.5 0.2 0.3\n")
