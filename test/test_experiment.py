from pathlib import Path
from unittest import TestCase

import shutil

import numpy
from sklearn.ensemble import RandomForestRegressor

from attribute_cancellation import AttributeCancellationTransformer
from experiment import run_experiment, get_final_experiment_path, plot_ranked_with_overfitting, get_train_test_scores, \
    create_subplots, plot_scores_splits, compare_models, plot_predictions, build_compare_models_result, plot_features_importance
from test.test_attribute_cancellation import generate_random_dataframe

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import scipy.stats

import matplotlib.pyplot as plt


class Test_experiments_methods(TestCase):
    def setUp(self) -> None:
        base_est = DecisionTreeRegressor()
        params = {"max_depth": scipy.stats.randint(5, 100)}

        self.act = AttributeCancellationTransformer(base_est, input_attribute='name_0', mode="ratio", params_distribution=params,
                                                    n_iter=5, cv=5,
                                                    shuffle=True, scoring="neg_mean_squared_error", random_state=2020)
        self.df = generate_random_dataframe(n_features=20, random_state=2020)

    def test_2_compare_models_from_dataframe(self):
        attributes = ["Immunoglobuline A (mg/dl)",
                      "Homocysteine (µmol/L)",
                      "Creatinina (mg/dl)"]
        groups = dict(RandomForest=[['baseline']])

        models_results = build_compare_models_result("results", groups)

        compare_models(highlight_attributes=attributes, models_results_df=models_results)

    def test_3_compare_models_from_file(self):
        attributes = ["Immunoglobuline A (mg/dl)",
                      "Homocysteine (µmol/L)",
                      "Creatinina (mg/dl)"]
        compare_models(highlight_attributes=attributes, load_cache_from_root='results')

    def test_4_compare_models_filter_experiments(self):
        attributes = ["Immunoglobuline A (mg/dl)",
                      "Homocysteine (µmol/L)",
                      "Creatinina (mg/dl)"]
        compare_models(highlight_attributes=attributes, load_cache_from_root='results', filter_experiment=attributes[:-1])

    def test_5_compare_models_all_attributes(self):
        compare_models(load_cache_from_root='results')

    def test_1_build_compare_models_result(self):
        models_scores = build_compare_models_result("results", dict(RandomForest=[['baseline']]))
        self.assertIsNotNone(models_scores)
        self.assertFalse(models_scores.empty)
        self.assertTrue(Path("results/models_scores.csv").exists())


class Test_get_final_experiment_path(TestCase):

    def test_subfolder_created(self):
        self.assertTrue(get_final_experiment_path("results", "RandomForest", ["depth", 'estimators'], False)
                        .exists())

    def test_subfolder_throw(self):
        self.assertRaises(FileNotFoundError, get_final_experiment_path, "results_invalid", "RandomForest",
                          ["depth", 'estimators'], True)


class TestPlots(TestCase):
    def setUp(self) -> None:
        base_est = RandomForestRegressor()
        params = {"max_depth": scipy.stats.randint(5, 6), "n_estimators": scipy.stats.randint(5, 10)}

        self.act = AttributeCancellationTransformer(base_est, input_attribute='name_0', mode="ratio", params_distribution=params,
                                                    n_iter=5, cv=5,
                                                    shuffle=True, scoring="neg_mean_squared_error", random_state=2020)
        self.df = generate_random_dataframe(n_features=10, random_state=2020)

    def tearDown(self) -> None:
        del self.act
        del self.df

    def test_create_subplots(self):
        fig, outer_axes = create_subplots(3, 3, 1, 2, figsize=(10, 10))

        for ax in outer_axes:
            ax[0, 0].annotate(xy=(0.5, 0.5), text="left")
            ax[0, 1].annotate(xy=(0.5, 0.5), text="right")
            ax[0, 1].remove()
        plt.show()

    def test_get_train_test_scores(self):
        df_new = self.act.fit_transform(self.df)
        train_test_scores = get_train_test_scores(self.act)
        print(train_test_scores.head().to_string())

    def test_plot_ranked_with_overfitting(self):
        df_new = self.act.fit_transform(self.df)
        for q in [4, 5, 10, 20]:
            with self.subTest(q=q):
                plot_ranked_with_overfitting(df_new, self.act, quantile=q, highlight_attributes=['name_3', 'name_5'])

    def test_plot_scores_splits(self):
        self.act.fit_transform(self.df)
        plot_scores_splits(self.act, highlight_attributes=['name_3', "name_5"])

    def test_plot_predictions(self):
        self.act.fit_transform(self.df)

        self.df.name_0 = self.df.name_0 + numpy.random.random(size=self.df.name_0.shape)
        plot_predictions(self.df, self.act, 'name_0', 'name_5')

    def test_plot_features_importance(self):
        df_new = self.act.fit_transform(self.df)
        plot_features_importance(self.df, df_new, 'name_9', ['name_1'], scoring="neg_mean_squared_error")
