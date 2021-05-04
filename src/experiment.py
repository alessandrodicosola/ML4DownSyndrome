import json
import math
import operator
import os
from collections import defaultdict
from itertools import product, groupby
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from joblib import Parallel, delayed
from matplotlib import gridspec
from matplotlib.patches import Patch
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from attribute_cancellation import AttributeCancellationTransformer, CrossValidation

import logger

log = logger.get_logger(__name__)


def save_dataframe(df: DataFrame, path):
    df.to_csv(path, index=False, sep=";")


def load_dataframe(path):
    return pd.read_csv(path, sep=";")


def validate_name(name: str):
    """Validate name removing chars that are non alphanumeric character, spaces, hyphen or rounded brackets"""
    import re
    new_name = re.sub(r'[^\w\s\-()]', '_', name)
    return new_name


_CONFIGURATION_FILENAME = "config.json"
_DATAFRAME_TRANSFORMED_FILENAME = "df_transformed.csv"
_ATTRIBUTE_CANCELLATION_TRANSFORMER_FILENAME = "act_data.joblib"
_MODELS_SCORES_FILENAME = "models_scores.csv"


def get_final_experiment_path(root: str, experiment: str, changes: list, throw_error: bool):
    """Returns the final path for the experiment as: {root}/{experiment}/{changes0}_{changes1}_{changesN}
    Args:
        root (str): Root folder for experiments
        experiment (str): Name of the current experiment
        changes (list): List of str of changes for which the experiment is done.
        throw_error (bool): If True will raise an error in case the final path doesn't exist otherwise it will create the folder.

    Returns:
        Path
    """
    # root folder where results are saved
    root = Path(root)
    # experiment folder
    experiment_name = experiment
    experiment = root / experiment
    # create a string concatenating all the strings in chagens with _
    sub_experiment = "_".join(changes)
    # create the folder for the sub experiment
    sub_experiment = experiment / sub_experiment
    if not sub_experiment.exists():
        if throw_error:
            raise FileNotFoundError(f"{sub_experiment} not found.")
        else:
            sub_experiment.mkdir(parents=True)

    return sub_experiment


def run_experiment(root: str,
                   experiment: str,
                   changes: list,
                   df_to_transform: DataFrame,
                   **config):
    """Run the experiment and save results

    Args:
        root (str): Root folder for experiments
        experiment (str): Name of the current experiment
        changes (list): List of str of changes for which the experiment is done
        df_to_transform (DataFrame): DataFrame preprocessed.
        **config: Config. See Keywords

    Keyword Args:
        mode (str): Discounting mode.
        input_attribute (str): Attribute to use as predictor for other attributes for discounting.
        base_estimator (estimator_object): Estimator to use for fitting and then discounting.
        params_distribution (dict): Dict of distribution or dict of lists that are sampled for trying n_iter combinations of parameters. See RandomSearchCV.
        n_iter (int): Number of parameters combinations to try.
        shuffle (bool): If True will shuffle data in each fold.
        cv (int): Number of splits for the KFold.
        scoring (str): Name of the scoring function to use for evaluation. Default: None (will use score function of the estimator)
        random_state (int): Set seed random state.
        n_jobs (int): Set number of jobs for cross validation.
    """
    experiment_path = get_final_experiment_path(root, experiment, changes, False)
    # display info message
    start_message = ["Experiment {} ( {} )".format(experiment, experiment_path),
                     "Changes:",
                     *[f"\t- {change:<20}" for change in changes],
                     "Configuration:\t",
                     *[f"\t- {key:<20}:\t{str(value)}" for key, value in config.items()]
                     ]
    log.info("\n".join(start_message))

    # get configs
    config_copy = {key: str(item) for key, item in config.items()}

    mode = config.pop('mode')
    input_attribute = config.pop('input_attribute')
    base_estimator = config.pop('base_estimator')
    params_distribution = config.pop('params_distribution')
    n_iter = config.pop('n_iter')
    shuffle = config.pop('shuffle')
    cv = config.pop('cv')
    scoring = config.get('scoring', None)
    random_state = config.get('random_state', None)
    n_jobs = config.pop('n_jobs')

    attribute_cancellation_transformer = AttributeCancellationTransformer(
        base_estimator=base_estimator,
        input_attribute=input_attribute,
        mode=mode,
        params_distribution=params_distribution,
        n_iter=n_iter,
        cv=cv,
        shuffle=shuffle,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs
    )

    log.info("Performing attribute cancellation...")
    df_transformed = attribute_cancellation_transformer.fit_transform(df_to_transform)

    log.info("Saving dataframe...")
    save_dataframe(df_transformed,
                   experiment_path / _DATAFRAME_TRANSFORMED_FILENAME)

    log.info("Saving transformer...")
    joblib.dump(attribute_cancellation_transformer,
                experiment_path / _ATTRIBUTE_CANCELLATION_TRANSFORMER_FILENAME)
    log.info("Saving configuration...")
    with open(experiment_path / _CONFIGURATION_FILENAME, 'w') as file:
        json.dump(config_copy, file)

    log.info("Experiment complete.")


def load_experiment(root: str, experiment: str, changes: list, df: bool = True, act: bool = True):
    """Load dataframe and attribute cancellation transformer of an experiment

    Args:
        root (str): Root folder for experiments
        experiment (str): Name of the current experiment
        changes (list): List of str of changes for which the experiment is done
        df (bool): If True will load the dataframe.
        act (bool): If True will load the attribute cancellation transformer.

    Returns:
        Tuple: (DataFrame transformed, AttributeCancellationTransformer fitted)
    """
    experiment_path = get_final_experiment_path(root, experiment, changes, True)

    log.info("Loading experiment {} ({})".format(experiment, experiment_path))

    dataframe_transformed = None
    attribute_cancellation_transformer = None

    if df:
        log.info("Loading dataframe transformed...")
        dataframe_transformed = load_dataframe(experiment_path / _DATAFRAME_TRANSFORMED_FILENAME)
    if act:
        log.info("Loading transformer...")
        attribute_cancellation_transformer = joblib.load(experiment_path / _ATTRIBUTE_CANCELLATION_TRANSFORMER_FILENAME)

    log.info("Loading complete.")

    return dataframe_transformed, attribute_cancellation_transformer


def get_ranked_dataframe_transformed(dataframe_transformed: DataFrame, act: AttributeCancellationTransformer) -> DataFrame:
    """Sort dataframe in ascending order by the mean absolute difference between discounted value and the reference value

    Args:
        dataframe_transformed (DataFrame): DataFrame discounted.
        act (AttributeCancellationTransformer):  AttributeCancellationTransformer fitted.

    Returns:
        DataFrame sorted
    """

    def mean_abs_diff(col):
        reference_value = act.mode_.reference
        return (col - reference_value).abs().mean()

    dataframe_transformed = dataframe_transformed.drop(columns=[act.input_attribute])
    return dataframe_transformed.agg([mean_abs_diff]).T.sort_values(by='mean_abs_diff', ascending=True)


def get_train_test_scores(act: AttributeCancellationTransformer):
    """
    Create a dataframe where rows are the attributes and columns are:

        -   mean_train_score
        -   std_train_score
        -   mean_test_score
        -   std_test_score
        -   mean_diff_test_train: score_val - score_train if maximize the score else error_val - error_train if minimize the score
        -   std_diff_test_train
    Args:
        act (AttributeCancellationTransformer): AttributeCancellationTransformer used for fitting the data

    Returns:
        DataFrame (n_attributes X [{mean,std}_{train,test}_score])
    """
    internal_list = list()
    for attribute in act.attribute_information_:
        cv_results: DataFrame = act.attribute_information_[attribute].cv_results

        # compute mean and std train and test score
        train_test_score = cv_results[['train_score', 'test_score']].agg(['mean', 'std'])
        d = {f"{prefix}_{suffix}": train_test_score.at[prefix, suffix] for prefix, suffix in
             product(['mean', 'std'], ['train_score', 'test_score'])}

        # compute the overfitting score
        # score_train - score_val if we have to maximize the score (e.g. accuracy)
        # error_val - error_train if we have to minimize the score that is maximize the negative score (e.e. MSE)

        sign = -1 if "error" in act.scoring else 1
        # score_train - score_val if the score is not an error
        # ( score_train - score_val) * -1 = score_val - score_train if the score is an error
        diff_train_test = (cv_results.train_score - cv_results.test_score) * sign

        mean_diff_test_train = diff_train_test.mean()
        std_diff_test_train = diff_train_test.std()
        d['mean_diff_test_train'] = mean_diff_test_train
        d['std_diff_test_train'] = std_diff_test_train

        # such that in the end i can do:
        # - score is not an error: score_train - score_val > tol
        # - score is an error: score_val - score_train > tol (that is error_val - error_train > tol )
        internal_list.append(d)

    return pd.DataFrame(data=internal_list, index=act.attribute_information_.keys())


def create_subplots(n_outer_rows: int, n_outer_cols: int, n_inner_rows: int, n_inner_columns: int,
                    figsize,
                    outer_wspace: float = 0.1,
                    outer_hspace: float = 0.1,
                    inner_sharex: bool = False,
                    inner_sharey: bool = False):
    """Create subplots with subplots

    Args:
        n_outer_rows (int): Number of columns
        n_outer_cols (int): Number of rows
        n_inner_rows (int): Number of rows in a single subplot
        n_inner_columns (int): Number of columns in a single subplot
        figsize (tuple): Figure size
        outer_wspace (float): Percentage of horizontal space between subplots
        outer_hspace (): Percentage of vertical space between subplots
        inner_sharex (): Each inner subplot will share x axis
        inner_sharey (): Each inner subplot will share y axis

    Examples:
        >>> fig, axes = create_subplots(n_outer_rows=3,n_outer_cols=3,n_inner_rows=2,n_inner_columns=1, figsize=(5,5), outer_wspace=0.1, outer_hspace=0.1, inner_sharex=True)
        >>> for ax in axes.flat:
        >>>     ax.plot(x,y)

    References:
        https://stackoverflow.com/questions/34933905/matplotlib-adding-subplots-to-a-subplot

        It doesn't contains this exactly code but only the idea.

    Returns:
        Figure, Axis (array with shape=(n_outer_rows * n_outer_cols, n_inner_rows, n_inner_columns))
    """
    # reference source:
    # axes to return
    # in order to be coherent with plt.subplots() use np.array
    axes = np.empty(shape=(n_outer_rows * n_outer_cols, n_inner_rows, n_inner_columns), dtype=np.object_)

    fig = plt.figure(figsize=figsize)
    # Create the grid that will contains plots of each attribute
    outer = gridspec.GridSpec(n_outer_rows, n_outer_cols, wspace=outer_wspace, hspace=outer_hspace, figure=fig)
    # Create for each attribute subplots inner subplots
    for outer_index in range(n_outer_rows * n_outer_cols):
        wspace = 0. if inner_sharey else 1.
        hspace = 0. if inner_sharex else 1.

        inner = gridspec.GridSpecFromSubplotSpec(n_inner_rows, n_inner_columns,
                                                 subplot_spec=outer[outer_index],
                                                 wspace=wspace, hspace=hspace)

        for row, col in product(range(n_inner_rows), range(n_inner_columns)):
            ax = plt.Subplot(fig, inner[row, col])
            axes[outer_index, row, col] = ax
            fig.add_subplot(ax)

    return fig, axes


def plot_ranked_with_overfitting(dataframe_transformed: DataFrame,
                                 act: AttributeCancellationTransformer,
                                 quantile: int,
                                 tol_overfitting: float = 50,
                                 highlight_overfitting=True,
                                 highlight_attributes=None):
    """Plot attributes ranked in ascending order by mean absolute difference between discounted value and reference value
    with mean train and test score in order to see which was overfitting.

    Args:
        dataframe_transformed (DataFrame): DataFrame discounted.
        act (AttributeCancellationTransformer): AttributeCancellationTransformer fitted.
        quantile (int): Quantile to use (e.g. 4 is [0.25, 0.5, 0.75, 1.0]). See pandas.qcut documentation.
        tol_overfitting (float): Tolerance for highlight overfitting ( There is overfitting if test - train > tol )
        highlight_overfitting (bool): If True highlight attributes that are overfitting
        highlight_attributes (list): List of str of attributes to highlight in the plot
    """
    # Get dataframe with mean train and test score
    train_test_scores = get_train_test_scores(act)

    # Get ranked dataframe
    ranked_df = get_ranked_dataframe_transformed(dataframe_transformed, act)
    assert ranked_df.mean_abs_diff.isna().sum() == 0

    # Create bins
    _KEY_INTERVAL = "bin"
    ranked_df[_KEY_INTERVAL] = pd.qcut(ranked_df.mean_abs_diff, quantile)
    assert ranked_df[_KEY_INTERVAL].isna().sum() == 0
    ranked_df_grouped = ranked_df.groupby(_KEY_INTERVAL)
    assert len(ranked_df_grouped) == quantile

    # Settings for plotting
    figsize = (20, 20)

    fontsize = 10

    n_rows, rest = quantile // 2, quantile % 2
    n_cols = 2 + rest
    outer_rows = n_rows
    outer_cols = n_cols
    inner_rows = 1
    # left: scores, right: diff w.r.t reference value
    inner_cols = 2

    diff_color = "#ffd3b4"
    highlight_color = "#98ddca"
    train_color = "#d5ecc2"
    test_color = "#ffaaa7"
    std_color = "#687980"

    fig, outer_axes = create_subplots(outer_rows, outer_cols, inner_rows, inner_cols, figsize=figsize,
                                      outer_wspace=0.5, outer_hspace=0.5,
                                      inner_sharey=True)

    for (interval, df_quantile), outer_ax in zip(ranked_df.groupby(_KEY_INTERVAL), outer_axes):
        # Set title
        title = f"Values between {interval}"

        if df_quantile.empty:
            # in order to remove empty subplots
            # this should work but it won't
            # outer_ax[0, 0].remove()
            # outer_ax[0, 1].remove()
            continue

        # Get axes reference
        ax_scores = outer_ax[0, 0]
        ax_diff = outer_ax[0, 1]

        # Get scores based on column names df_quantile (used as index)
        scores = train_test_scores.loc[df_quantile.index, :]
        mean_values = scores[['mean_train_score', 'mean_test_score']]
        # in order to have error bar the std_values should contains column names equal to the names in mean_values
        std_values = scores[['std_train_score', 'std_test_score']].rename(columns=dict(
            std_train_score="mean_train_score",
            std_test_score="mean_test_score"
        ))

        # Set colors based on if the attribute is overfitting or not
        scores['color'] = diff_color
        if highlight_overfitting:
            # where diff is less than toll than is used diff_color otherwise highlight_color
            scores.color.where(scores.mean_diff_test_train <= tol_overfitting, other=highlight_color, inplace=True)

        # Plot differences with respect the reference value
        df_quantile.mean_abs_diff.plot.barh(ax=ax_diff,
                                            fontsize=fontsize,
                                            align='center',
                                            color=scores.color)
        # Plot mean and std of test scores
        mean_values.plot.barh(ax=ax_scores, fontsize=fontsize, color=[train_color, test_color],
                              xerr=std_values, capsize=2, ecolor=std_color)

        # Set title
        ax_diff.set_title(title)
        # # invert x axis in order to have mirrored plots
        ax_scores.invert_xaxis()
        # invert y axis in both since pandas will plot in ascending order starting from bottom to top
        ax_diff.invert_yaxis()
        ax_scores.invert_yaxis()
        # hide y labels in the diff plot
        ax_diff.get_yaxis().set_visible(False)
        # Set color attributes to highlight
        if highlight_attributes:
            for label in ax_scores.get_yticklabels():
                if label.get_text() in highlight_attributes:
                    label.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))

    handles = [Patch(color='red', label="Attributes highlighted"),
               Patch(color=diff_color, label=f"Mean of absolute differences between discounted values and reference value ({act.mode_.reference})"),
               Patch(color=highlight_color, label="Attributes that are overfitting")]
    loc = 'center'
    fig.legend(handles=handles, loc=loc)

    plt.show()


def plot_scores_splits(act: AttributeCancellationTransformer, highlight_attributes: list = None):
    """Plot the scores for each split. Different band of color in the background of each axis represent a different combination
    of parameters.

    Args:
        act (AttributeCancellationTransformer): AttributeCancellationTransformer fitted
        highlight_attributes (list): if None display the scores for all attributes otherwise only the ones in the list
    """
    # config
    train_color = "#d5ecc2"
    test_color = "#ffaaa7"
    span_color1 = "#d9e4dd"
    span_color2 = "#fbf7f0"
    span_colors = [span_color1, span_color2]

    attributes_to_search = act.attribute_information_ if not highlight_attributes else highlight_attributes

    total = len(attributes_to_search)
    n_row, n_col = get_rows_cols_subplots(total)

    fig, axes = plt.subplots(n_row, n_col, figsize=(20, 20))

    for index, (attribute, ax) in enumerate(zip(attributes_to_search, axes.flat)):
        cv_results = act.attribute_information_[attribute].cv_results[['index_split', 'train_score', 'test_score']]
        cv_results.plot.bar(ax=ax, x='index_split', y=['train_score', 'test_score'], color=[train_color, test_color])
        ax.set_title(attribute)

        start, end = -1, act.cv - 1
        for iteration in range(act.n_iter):
            ax.axvspan(start, end, facecolor=span_colors[iteration % len(span_colors)], alpha=0.3, zorder=-1)
            start = end
            end = end + act.cv

    title = "Train and test score per fold in each iteration"
    if act.scoring:
        title += f"({act.scoring})"
    fig.suptitle(title)

    # create legend
    handles = [ Patch(facecolor=span_color1, edgecolor=span_color2, hatch=r"\\",
                      label="Each band represent a different combination of parameters")]
    fig.legend(handles=handles, fontsize=18)
    fig.tight_layout()
    plt.show()


def build_compare_models_result(root: str, groups: dict):
    """Aggregate all results from all models within single file

    Args:
        root (str): Results folder.
        groups (dict): Dictionary of { experiment_name: [ ['change1', 'change2'], ['change3'], ... ] }.

    Returns:
        DataFrame
    """
    path_to_save = Path(root)
    path_to_save = path_to_save / _MODELS_SCORES_FILENAME

    if path_to_save.exists():
        os.remove(path_to_save)

    get_key_fn = operator.itemgetter(0)
    # sort groups by model name
    groups = sorted(groups.items(), key=get_key_fn)
    groups = groupby(groups, key=get_key_fn)

    models_scores_df: DataFrame = pd.DataFrame()

    def single_model_scores(model_name, changes):
        df, act = load_experiment(root, model_name, changes)
        current_model = get_train_test_scores(act)
        current_model['model_name'] = model_name + "_" + "_".join(changes)
        current_model['attribute'] = current_model.index.values
        return current_model

    with Parallel(n_jobs=-1) as parallel:
        models_scores = parallel(delayed(single_model_scores)(model_name, changes)
                                 for model_name, group in groups
                                 for _, list_of_changes in group
                                 for changes in list_of_changes)

        for model_scores in models_scores:
            models_scores_df = models_scores_df.append(model_scores)

        save_dataframe(models_scores_df, path_to_save)
        return models_scores_df


def compare_models(highlight_attributes: list = None, filter_experiment: list = None,
                   models_results_df: DataFrame = None,
                   load_cache_from_root: str = None):
    """ Compare scores among all the models for specified attributes

    Args:
        highlight_attributes (list): List of attributes to show.
        filter_experiment (list): if not None experiment in filter_experiment are not shown.
        models_results_df (DataFrame): if not None will be loaded.
        load_cache_from_root (str): Root results folder. If not None will be loaded the csv in this folder.
    """
    # assert is true if only one of the two options are true
    assert (models_results_df is not None) ^ (load_cache_from_root is not None)
    if not highlight_attributes:
        log.warning("Without selecting specific attributes the plot will be generated slowly and it will be overcrowded.")
    if load_cache_from_root:
        load_cache = Path(load_cache_from_root) / _MODELS_SCORES_FILENAME
        models_scores = load_dataframe(load_cache)
    else:
        models_scores = models_results_df

    models_scores = models_scores.set_index('attribute')

    if highlight_attributes:
        models_scores = models_scores.loc[highlight_attributes, :]
        total = len(highlight_attributes)
    else:
        total = len(models_scores.index)

    # config
    train_color = "#d5ecc2"
    test_color = "#ffaaa7"
    std_color = "#687980"
    colors = [train_color, test_color]

    n_row, n_col = get_rows_cols_subplots(total)
    fontsize = 15

    fig, axes = plt.subplots(n_row, n_col, figsize=(20, 20), sharey='all')

    map_columns = {
        "std_train_score": 'mean_train_score',
        "std_test_score": "mean_test_score"
    }
    for (attribute, df_attribute), ax in zip(models_scores.groupby('attribute'), axes.flat):
        df_attribute = df_attribute.set_index('model_name')
        if filter_experiment:
            # ~ isin == not is in
            index_to_keep = ~df_attribute.index.isin(filter_experiment)
        else:
            index_to_keep = df_attribute.index

        df_mean = df_attribute.loc[index_to_keep, ['mean_train_score', 'mean_test_score']]
        # df_std must have same column names in order to xerr to work.
        df_std = df_attribute.loc[index_to_keep, ['std_train_score', 'std_test_score']].rename(columns=map_columns)
        df_mean.plot.barh(ax=ax, xerr=df_std, fontsize=fontsize, color=colors, ecolor=std_color, capsize=2)
        ax.set_title(attribute)

    fig.tight_layout()
    plt.show()


def plot_features_importance(df_original, df_discounted, target_attribute: str, excluded_attributes: list, scoring: str,
                             max_features: int = 20):
    """ Fit RandomForestRegressor and RandomForestRegressor (using CrossValidation) on the original and discounted dataset and then plot the features most important used
    by the models.

    In order to have a meaningful comparison random_state is set inside in order to use same parameters and same split.

    Target attribute and excluded attributes are removed from both the original and discounted dataset.

    Args:
        df_original (DataFrame): original dataset.
        df_discounted (DataFrame): discounted dataset with AttributeCancellationTransformer.
        target_attribute (str): Target attribute for the regression
        excluded_attributes (list): List of attributes to exclude from both the datasets
        scoring (str): scoring function to use
        max_features (int): The number of important features to plot. Default: 20.
    """
    # Get target
    y_original = df_original[target_attribute]
    y_transformed = df_discounted[target_attribute]

    # Remove input attribute and any others
    X_original = df_original.drop(columns=excluded_attributes + [target_attribute])
    X_transformed = df_discounted.drop(columns=excluded_attributes + [target_attribute])

    # Define data
    data = dict(original=(X_original, y_original), discounted=(X_transformed, y_transformed))

    # Models configuration
    models = [RandomForestRegressor(n_jobs=-1), ExtraTreesRegressor(n_jobs=-1)]
    params_distribution = dict(max_depth=scipy.stats.randint(10, 50),
                               n_estimators=scipy.stats.randint(50, 100),
                               min_samples_split=scipy.stats.randint(20, 30),
                               min_samples_leaf=scipy.stats.randint(20, 30))
    fixed_random_state = 2020

    # Collect features selected by both same model or same dataset
    features_selected = defaultdict(list)

    # Plot configuration
    original_color = "#ffd3b4"
    discounted_color = "#98ddca"
    model_label_color = "#8675a9"
    kind_label_color = "#e36387"
    bar_colors = dict(original=original_color, discounted=discounted_color)
    # Set for original and discounted features selected a color and for randomforest and extratrees another one
    kinds = ['original', 'discounted']
    label_colors = dict(**{kind: kind_label_color for kind in kinds},
                        **{model.__class__.__name__: model_label_color for model in models})

    fontsize = 14
    fontsize_legend = 18

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

    for (model, kind), ax in zip(product(models, kinds), axes.flat):
        X, y = data[kind]
        cv = CrossValidation(base_estimator=model,
                             params_distribution=params_distribution,
                             n_iter=5,
                             cv=5,
                             shuffle=True,
                             n_jobs=-1,
                             refit=True,
                             scoring=scoring,
                             random_state=fixed_random_state).fit(X, y)
        feature_importances = pd.Series(data=cv.best_estimator_.feature_importances_, index=X.columns).sort_values(
            ascending=False).head(max_features)
        feature_importances.plot.barh(ax=ax, color=bar_colors[kind], fontsize=fontsize)
        # invert y axis since plot.barh draw from bottom to top
        ax.invert_yaxis()

        # set title with the kind and scores
        train_score = f"{cv.best_mean_train_score_:.3f} (+/- {cv.best_std_train_score_:.3f})"
        test_score = f"{cv.best_mean_test_score_:.3f} (+/- {cv.best_std_test_score_:.3f})"
        title = '\n'.join([kind, f"train_score: {train_score}", f"test_score: {test_score}"])
        ax.set_title(title, fontsize=fontsize)

        # set the xlabel with the model
        ax.set_xlabel(cv.best_estimator_, fontsize=fontsize)

        features_selected[model.__class__.__name__].append(feature_importances.index.to_list())
        features_selected[kind].append(feature_importances.index.to_list())

    features_selected = dict(map(get_intersection, features_selected.items()))

    for (model, kind), ax in zip(product(models, kinds), axes.flat):
        model_name = model.__class__.__name__
        log.debug("%s", model_name)
        for label in ax.get_yticklabels():
            if label.get_text() in features_selected[kind]:
                label.set_bbox(dict(facecolor=label_colors[kind], alpha=0.2, edgecolor=label_colors[kind]))
            if label.get_text() in features_selected[model_name]:
                label.set_bbox(dict(facecolor=label_colors[model_name], alpha=0.2, edgecolor=label_colors[model_name]))

    # Add information about colors
    patch_mode = Patch(color=kind_label_color, label="Attributes in same dataset")
    patch_model = Patch(color=model_label_color, label="Attributes in same model")
    patch_original_features_importance = Patch(color=original_color, label="Features importance on the original dataset")
    patch_discounted_features_importance = Patch(color=discounted_color, label="Features importance on the discounted dataset")
    handles = [patch_original_features_importance, patch_discounted_features_importance, patch_mode, patch_model]
    loc = (.70, .55)
    fig.legend(handles=handles, loc=loc, fontsize=fontsize_legend)

    fig.tight_layout()
    plt.show()


def get_intersection(elem):
    kind, list_of_features = elem
    out_set = set(list_of_features[0])

    for l in list_of_features:
        out_set &= set(l)

    return kind, out_set


def plot_predictions(df_original: DataFrame, act: AttributeCancellationTransformer, input_attribute: str, target_attribute: str):
    true_color = "#ffd3b4"
    pred_color = "#98ddca"

    input = df_original[input_attribute].values.reshape(-1, 1)
    target = df_original[target_attribute]
    predictions = act.attribute_information_[target_attribute].estimator.predict(input)

    fig, ax = plt.subplots()
    ax.scatter(input, target, color=true_color, label='True values')
    ax.scatter(input, predictions, color=pred_color, label='Predicted values')
    ax.set_xlabel(input_attribute)
    ax.set_ylabel(target_attribute)
    fig.legend()
    plt.show()


def get_rows_cols_subplots(total):
    """ Get rows and cols for subplots by sqrt of total

    Returns
        n_row, n_col
    """
    n_row = math.ceil(math.sqrt(total))
    n_col = total // n_row + (0 if total % n_row == 0 else 1)
    return n_row, n_col
