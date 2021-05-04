import sys
from unittest import TestCase

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from scipy.stats import randint
from pprint import pprint

import numpy as np
import pandas as pd

from attribute_cancellation import CrossValidation, AttributeCancellationTransformer

random_state = 2020


def generate_random_dataframe(n_samples=100, n_features=10, format_column="name_{}", random_state=None):
    """Generate a random dataframe naming column using the format specified.
    The name should contains exactly one pair of brackets.

    Examples:
        df = generate_random_dataframe(format_column="column_{}")

        df = generate_random_dataframe(format_column="{}")

        df = generate_random_dataframe(format_column="the_{}_th_column")

    Args:
        n_samples ():
        n_features ():
        format_column ():

    Returns:

    """
    if random_state:
        np.random.seed(random_state)
    columns = [format_column.format(i) for i in range(n_features)]
    values = [[np.random.rand() for i in range(n_samples)] for _ in range(n_features)]
    return pd.DataFrame(dict(zip(columns, values)))


class TestCrossValidation(TestCase):

    def setUp(self) -> None:

        base_estimator = RandomForestRegressor()
        params_distribution = dict(
            n_estimators=randint(50, 100),
            max_depth=randint(5, 10),
            min_samples_split=randint(5, 10)
        )

        self.cv = CrossValidation(base_estimator=base_estimator,
                                  params_distribution=params_distribution,
                                  cv=5,
                                  n_iter=5,
                                  shuffle=True,
                                  scoring="neg_mean_squared_error",
                                  random_state=random_state)

        self.X, self.y = make_regression()

    def tearDown(self) -> None:
        del self.cv

    def test_randomness(self):
        self.assertTrue(self.cv.base_estimator.random_state == random_state)
        self.assertTrue(self.cv.cv.random_state == random_state)
        self.cv.fit(self.X, self.y)
        self.assertTrue(self.cv.best_estimator_.random_state == random_state)


    def test__fit_no_fail(self):
        try:
            outs = self.cv._fit(self.X, self.y)
        except Exception as ex:
            self.fail(ex)

    def test_fit_no_fail(self):
        try:
            outs = self.cv.fit(self.X, self.y)
        except Exception as ex:
            self.fail(ex)

    def test_cv_has_objs_after_fit(self):
        self.cv.fit(self.X, self.y)

        for attr in ['cv_results_', 'best_estimator_', 'best_params_', "best_mean_test_score_", "best_std_test_score_",
                     "best_mean_train_score_",
                     "best_std_train_score_"]:
            with self.subTest(attr=attr):
                self.assertTrue(hasattr(self.cv, attr))
                self.assertIsNotNone(getattr(self.cv, attr))

    def test_score(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        self.cv.fit(X_train, y_train)
        try:
            self.cv.score(X_test, y_test)
        except:
            self.fail()

    def test_predict(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        self.cv.fit(X_train, y_train)
        try:
            self.cv.predict(X_test)
        except:
            self.fail()


class TestAttributeCancellationTransformer(TestCase):

    def setUp(self) -> None:
        base_estimator = make_pipeline(StandardScaler(), MLPRegressor())
        params_distribution = dict(
            mlpregressor__hidden_layer_sizes=randint(5, 10),
        )

        base_estimator = DecisionTreeRegressor()
        params_distribution = dict(
            max_depth=randint(5, 10),
            min_samples_split=randint(5, 10)
        )

        base_estimator = RandomForestRegressor()
        params_distribution = dict(
            n_estimators=randint(50, 100),
            max_depth=randint(5, 10),
            min_samples_split=randint(5, 10)
        )

        self.input_attribute = "name_0"
        self.target_attribute = "name_1"
        self.df = generate_random_dataframe(n_samples=200, n_features=5, format_column="name_{}")
        self.act = AttributeCancellationTransformer(base_estimator, self.input_attribute, mode='ratio',
                                                    params_distribution=params_distribution,
                                                    n_iter=1, cv=5, shuffle=True, scoring="neg_mean_squared_error",
                                                    random_state=None, n_jobs=1)
        self.columns = self.df.columns.drop(self.input_attribute)

    def tearDown(self) -> None:
        del self.act
        del self.df

    def test_fit(self):
        try:
            self.act.fit(self.df)
        except:
            self.fail("Error in fit")

    def test_transform(self):
        try:
            self.act.fit_transform(self.df)
        except:
            self.fail("Error in fit")

    def test_has_attribute_information(self):
        try:
            self.act.fit(self.df)
            self.assertTrue(hasattr(self.act, 'attribute_information_'))
            self.assertIsNotNone(self.act.attribute_information_)
            for column in self.columns:
                with self.subTest(column=column):
                    self.assertIsNotNone(self.act.attribute_information_[column].cv_results)
                    self.assertIsNotNone(self.act.attribute_information_[column].estimator)
                    pprint(self.act.attribute_information_[column].cv_results)
        except:
            self.fail()
