# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import os

import numpy as np
import pandas as pd
import openml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from dwave.plugins.sklearn.transformers import SelectFromQuadraticModel


class DataSetBase:
    """Base class for datasets.

    Subclasses should define the following attributes:
        X (array): Feature data with features as columns.
        y (array): Target data.
        n (int): Number of features.
        baseline_cv_score (float):
            Baseline cross-validation score with all features.
        score_range (tuple):
            Lower and upper values for displaying cross-validation scores.
        default_redundancy_penalty (float)
        default_k (int): Default setting for number of features to select.
    """

    def get_relevance(self):
        """Return array of values for relevance of each feature to the target."""
        return np.array([abs(np.corrcoef(x, self.y)[0, 1]) for x in self.X.values.T])

    def calc_redundancy(self):
        """Compute and return 2d array of feature redundancy values."""
        return abs(np.corrcoef(self.X.values, rowvar=False))

    def get_redundancy(self):
        """Return 2d array of feature redundancy values, possibly cached to disk."""
        # The following logic can be used to store the redundancy matrix to disk
        # so that it does not need to be computed each time the app is launched.
        # This is probably not needed for dataset sizes that would be used with
        # the app, as np.corrcoef is quite fast.
        if self.n > 500:
            data_path = f"redundancy-{self.name}.pkl"
            if os.path.exists(data_path):
                return pickle.load(open(data_path, "rb"))
            else:
                print("Calculating redundancy data...")
                data = self.calc_redundancy()
                print("Storing redundancy data")
                with open(data_path, "wb") as f:
                    pickle.dump(data, f)
                return data
        else:
            return self.calc_redundancy()

    def get_selected_features(self, X_new):
        """Post-processes result from plug-in to return features

         Args:
            X_new (np.ndarray):
                Reduced dataset with selected features

        Returns:
            Array of indices of selected features.
        """
        _, n = self.X.shape
        _, m = X_new.shape
        # need to iterate through and enumerate which features were selected
        feature_names = []
        for i in range(n):
            for j in range(m):
                if np.all(X_new[:, j] == self.X.iloc[:, i]):
                    feature_names.append(i)
                    break
        return feature_names

    def solve_feature_selection(self, k, alpha, time, solver):
        """Construct and solve feature selection using plugin.

        Args:
            k (int):
                Number of features to select.
            alpha (float):
                Parameter between 0 and 1 that defines the relative weight of
                linear and quadratic coefficients.
            time (int):
                Number of seconds for solver to run.
            solver (str):
                String dictating use of either CQM or NL.

        Returns:
            Array of indices of selected features.
        """

        X_new = SelectFromQuadraticModel(
            num_features=k, alpha=alpha, time_limit=time, solver=solver
        ).fit_transform(self.X.values, self.y)
        return self.get_selected_features(X_new)

    def score_indices_cv(self, indices, cv=3):
        """Compute the accuracy score of a random forest classifier trained using the specified
        features.

        Args:
            indices (array): Array of feature indices
            cv (int): Number of folds.

        Returns:
            float: Cross-validation accuracy score.
        """
        clf = RandomForestClassifier()

        # NB: the classifier is both trained and evaluated `cv` times in a loop inside
        # NB: this call to `sklearn.model_selection.cross_val_score`:
        return np.mean(cross_val_score(clf, self.X.iloc[:, indices], self.y, cv=cv))

    def score_baseline_cv(self, reps=5):
        """Compute baseline accuracy score of a random forest classifier trained using all
        features.

        Args:
            reps (int): Number of times to repeat cross-validation.

        Returns:
            float: Average cross-validation accuracy score across `nreps` repetitions.
        """
        indices = list(range(np.size(self.X, 1)))
        return np.mean([self.score_indices_cv(indices) for i in range(reps)])


class Titanic(DataSetBase):
    def __init__(self):
        df = pd.read_csv("formatted_titanic.csv")
        target_col = "survived"
        self.X = df.drop(target_col, axis=1).astype(float)
        self.y = df[target_col].values
        self.baseline_cv_score = 0.69

        self.score_range = (0.6, 0.8)

        self.default_redundancy_penalty = 0.68
        self.default_k = 8

        self.n = np.size(self.X, 1)
        self.name = "titanic"


class Scene(DataSetBase):
    def __init__(self):
        data_id = 312
        self.baseline_cv_score = 0.90

        dataset = openml.datasets.get_dataset(data_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        self.y = y.values.astype(int)
        X = X.astype(float)
        self.X = X

        self.score_range = (0.79, 0.95)
        self.default_k = 30
        self.default_redundancy_penalty = 0.4

        self.n = np.size(self.X, 1)
        self.name = "scene"


def DataSet(name):
    """Return instance of specified DataSet class.

    Args:
        name (str)
            Name of feature selection dataset: either 'titanic' or 'scene'.

            The 'titanic' dataset contains 14 features, and the 'scene' dataset
            contains 299 features.
    """
    datasets = {"titanic": Titanic, "scene": Scene}
    return datasets[name]()


if __name__ == "__main__":
    # Compute baseline scores, which are stored as part of the DataSet
    # definitions.

    dataset_names = ("titanic", "scene")

    for name in dataset_names:
        score = DataSet(name).score_baseline_cv()
        print(f"Baseline score for {name}: {score}")
