class BaseSolver(object):
    def get_state_dict(self):
        """
        Returns
        -------
        state_dict: dict
            Dict containing values of members and other solver-specific
            things to be saved.
        """
        raise NotImplementedError

    def load_from_state_dict(self, state_dict):
        """
        Given a state dict, load the model's state.

        Parameters
        ----------
        state_dict: dict
            Dict containing saved values of members and other solver-specific
            things to be loaded.
        """
        raise NotImplementedError

    def load_from_file(self, load_path):
        """
        Parameters
        ----------
        load_path: str
            Path to the model's serialized state to load.

        Returns
        -------
        self:
            The current object that had its state loaded.
        """
        raise NotImplementedError

    def read_data(self):
        """
        Given paths to data, process and read them accordingly.
        """
        raise NotImplementedError

    def save_to_file(self, save_dir, run_id):
        """
        Parameters
        ----------
        save_dir: str
            The folder to save the trained model in.

        run_id: str
            The run_id for this model, also dictates the serialization filename.
        """
        raise NotImplementedError

    def train_model(self):
        """
        Train the model.
        """
        raise NotImplementedError

    def translate_file(self, oov_path, show_progbar=True, n_jobs=1):
        """
        Given a file, predict translations for each data example
        (line of file).

        Parameters
        ----------
        oov_path: str
            Path to file with OOV data. Different subclasses
            may have different preferences for the sort of data
            they use.

        show_progbar: boolean, optional (default=True)
            Whether or not to show a progress bar.

        n_jobs: int, optional (default=1)
            The number of processes to spawn for translating in parallel,
            if supported by the solver.
        """
        raise NotImplementedError

    def translate_list(self, oov_list, show_progbar=True, n_jobs=1, debug=False):
        """
        Given a list of OOV words, predict translations for each of them.

        Parameters
        ----------
        oov_list: List of str
            List of OOV strings to translate.

        debug: boolean, optional (default=False)
            Whether or not to print useful information about the model's
            prediction process. This is mainly used when using the model
            in interactive mode in an attempt to understand the model better.

        Returns
        -------
        predicted_translation_list: List of str
            List of predicted translations.

        show_progbar: boolean, optional (default=True)
            Whether or not to show a progress bar.

        n_jobs: int, optional (default=1)
            The number of processes to spawn for translating in parallel,
            if supported by the solver.
        """
        raise NotImplementedError
