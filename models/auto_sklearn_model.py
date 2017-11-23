# -*- encoding: utf-8 -*-
import os
import sys
curDir = os.path.dirname(__file__)
sys.path.append('{0}/../scripts/'.format(curDir))
import multiprocessing
import shutil
from model import Model
import sklearn.model_selection
import sklearn.metrics
from sklearn.model_selection import train_test_split
from autosklearn.metrics import accuracy
from autosklearn.classification import AutoSklearnClassifier

class AutoSklearnModel(Model):

    # Set how many processes to run to execute autosklearn
    NUM_PROCESSES = 16

    TMP_FOLDER = '/tmp/autosklearn_parallel_example_tmp'
    OUTPUT_FOLDER = '/tmp/autosklearn_parallel_example_out'

    model = None

    X_train = None
    X_test = None
    Y_train = None
    Y_test = None


    def spawn_classifier(self, seed):
        """Spawn a subprocess.

        auto-sklearn does not take care of spawning worker processes. This
        function, which is called several times in the main block is a new
        process which runs one instance of auto-sklearn.
        """

        # Use the initial configurations from meta-learning only in one out of
        # the four processes spawned. This prevents auto-sklearn from evaluating
        # the same configurations in four processes.
        if seed == 0:
            initial_configurations_via_metalearning = 25
        else:
            initial_configurations_via_metalearning = 0

        # Arguments which are different to other runs of auto-sklearn:
        # 1. all classifiers write to the same output directory
        # 2. shared_mode is set to True, this enables sharing of data between
        # models.
        # 3. all instances of the AutoSklearnClassifier must have a different seed!
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60, # sec., how long should this seed fit
            # process run
            per_run_time_limit=15, # sec., each model may only take this long before it's killed
            # ml_memory_limit=1024, # MB, memory limit imposed on each call to a ML algorithm
            shared_mode=True, # tmp folder will be shared between seeds
            tmp_folder=self.TMP_FOLDER,
            output_folder=self.OUTPUT_FOLDER,
            delete_tmp_folder_after_terminate=False,
            delete_output_folder_after_terminate=False,
            ensemble_size=0, # ensembles will be built when all optimization runs are finished
            initial_configurations_via_metalearning=initial_configurations_via_metalearning,
            seed=seed)
        automl.fit(self.X_train, self.Y_train)

    def train(self, X_df, Y):

        return None

    def predict(self, X_df):

        return None

    def get_model(self, X, Y):
        for folder in [self.TMP_FOLDER, self.OUTPUT_FOLDER]:
            try:
                shutil.rmtree(folder)
            except OSError:
                pass

        # Train/Test Split
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(X, Y, test_size=0.30, random_state=42)

        # Start processes
        processes = []
        for i in range(self.NUM_PROCESSES): # set this at roughly half of your cores
            p = multiprocessing.Process(target=self.spawn_classifier, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        print('Starting to build an ensemble!')
        automl = AutoSklearnClassifier(time_left_for_this_task=15,
                                       per_run_time_limit=15,
                                       shared_mode=True,
                                       ensemble_size=50,
                                       ensemble_nbest=200,
                                       tmp_folder=self.TMP_FOLDER,
                                       output_folder=self.OUTPUT_FOLDER,
                                       initial_configurations_via_metalearning=0,
                                       seed=1)

        # Both the ensemble_size and ensemble_nbest parameters can be changed now if
        # necessary
        automl.fit_ensemble(self.Y_train,
                            task=1,
                            metric=accuracy,
                            precision='32',
                            ensemble_size=20,
                            ensemble_nbest=50)

        predictions = automl.predict(self.X_test)
        score = sklearn.metrics.accuracy_score(self.Y_test, predictions)
        print(automl.show_models())

        return (automl, score)
