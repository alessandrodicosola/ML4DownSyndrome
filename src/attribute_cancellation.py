from collections import namedtuple
from functools import partial
from itertools import product
from time import perf_counter

import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame, Series
from sklearn.base import clone, TransformerMixin, BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import ParameterSampler, KFold
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

# for progress
try:
    from tqdm.auto import tqdm

    IS_TQDM_INSTALLED = True
except:
    IS_TQDM_INSTALLED = False
import logging

# get logger
log = logging.getLogger(__name__)

# constant for operation where axis= is used
_OPERATION_COLUMN_WISE = 0
_OPERATION_ROW_WISE = 1


class _measure_time:
    """Class for measuring time of execution using 'with' keyword

    Examples:
        >>>with _measure_time() as me:
        >>>     do_something()
        >>>print(me.elapsed.seconds)
        >>>print(me.elapsed.minutes)
        >>>print(me.elapsed.hours)
    """

    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        elapsed_seconds = perf_counter() - self.start
        self.elapsed = Bunch(seconds=elapsed_seconds, minutes=elapsed_seconds // 60, hours=elapsed_seconds // 3600)


def validate_estimator(estimator, random_state=None):
    """Check that estimator contains 'fit' and 'predict' and then set the random state

    Args:
        estimator: estimator to check.
        random_state (int): seed to use as random state.

    Returns:
        estimator checked
    """
    estimator = clone(estimator)
    base_msg = "%s doesn't implement %s"
    base_name = estimator.__class__.__name__
    assert hasattr(estimator, 'fit'), base_msg.format(base_name, "fit")
    assert hasattr(estimator, 'predict'), base_msg.format(base_name, "predict")

    # Add random state
    if random_state:
        if hasattr(estimator, 'random_state'):
            setattr(estimator, 'random_state', random_state)

    return estimator


class CrossValidation:
    """CrossValidation tunes an estimator sampling parameters randomly.

    It contains 'predict', 'score' function which will use the attribute best_estimator_.

    The attribute cv_results_ is a DataFrame with the columns:
                - estimator
                - train_score
                - test_score
                - train_time
                - inference_time
                - score_time
                - test_indices
                - test_predictions
                - index_params
                - params
                - index_split

    Args:
        base_estimator (estimator object): Estimator to fit using CrossValidation.
        params_distribution (dict): Parameters distribution using scipy or list.
        n_iter (int): Number of combination to try.
        cv: Number of splits for each k-fold.
        shuffle (bool): If True shuffle data in each fold.
        pre_dispatch: Controls the number of jobs that get dispatched during parallel execution. See GridSearch documentation.
        n_jobs (int): Number of jobs for the parallel execution.
        refit (bool): If True refit the estimator on the whole dataset otherwise takes the one whose test score is the best.
        scoring (str): Name of the score to use.
        random_state (int): Seed for initialize random state.

    Attributes:
        best_estimator_
        best_params_
        best_mean_test_score_
        best_std_test_score_
        cv_results_ = None
    """

    def __init__(self, base_estimator,
                 params_distribution: dict,

                 n_iter: int,
                 cv,
                 shuffle: bool,

                 scoring: str,

                 pre_dispatch='n_jobs',
                 n_jobs: int = 1,
                 refit: bool = True,

                 random_state=None):

        self.cv = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)

        self.base_estimator = validate_estimator(estimator=base_estimator, random_state=random_state)

        self.params_distribution = params_distribution

        self.n_iter = n_iter

        self.scoring = scoring

        self.n_jobs = n_jobs
        self.refit = refit
        self.pre_dispatch = pre_dispatch

        self.random_state = random_state

        # Set after fit
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_mean_test_score_ = None
        self.best_std_test_score_ = None
        self.best_mean_train_score_ = None
        self.best_std_train_score_ = None
        self.cv_results_ = None

    def _clone_base_estimator_with_params(self, parameters=None):
        """Clone an estimator object (copying its parameter) without fitting data.

        Args:
            parameters (dict): In None the setimator will use the specified parameters.

        Returns:
            Estimator with specified parameters
        """
        current_estimator = clone(self.base_estimator)

        # Get parameters
        # taken directly from sklearn source
        # https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/model_selection/_validation.py#L452
        if parameters is not None:
            # clone after setting parameters in case any parameters
            # are estimators (like pipeline steps)
            # because pipeline doesn't clone steps in fit
            cloned_parameters = {}
            for k, v in parameters.items():
                cloned_parameters[k] = clone(v, safe=False)

            current_estimator = current_estimator.set_params(**cloned_parameters)

        return current_estimator

    def _fit_and_score(self, X, y, train_indices, test_indices, parameters, index_params, index_split, **fit_params):
        """ Fit a single estimator using specified parameters, on the training set ( specified by train_indices ).
        Compute predictions for both train fold and validation fold.

        Args:
            X (np.array): Inputs
            y (np.array): Targets
            train_indices (np.array): array of indices computed by KFold
            test_indices (np.array): array of indices computed by KFold
            parameters (dict): parameters for the estimator
            index_params (int): index of the parameters among all the combination sampled.
            index_split (int): split of the index in the current kfold
            **fit_params: additional parameters

        Returns:
            Dictionary with:
                - estimator
                - train_score
                - test_score
                - train_time
                - inference_time
                - score_time
                - test_indices
                - test_predictions
                - index_params
                - params
                - index_split
        """

        # Get train and test split
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # Get estimator
        current_estimator = self._clone_base_estimator_with_params(parameters)

        # Fit the estimator collecting time
        with _measure_time() as train_time:
            current_estimator = current_estimator.fit(X_train, y_train, **fit_params)

        # Make predictions
        with _measure_time() as inference_time:
            test_predictions = current_estimator.predict(X_test)
            train_predictions = current_estimator.predict(X_train)

        # Evaluate predictions
        with _measure_time() as score_time:
            train_score = self.compute_score(y_true=y_train, y_preds=train_predictions)
            test_score = self.compute_score(y_true=y_test, y_preds=test_predictions)

        out = dict(
            estimator=current_estimator,

            train_score=train_score,
            test_score=test_score,

            train_time=train_time.elapsed,
            inference_time=inference_time.elapsed,
            score_time=score_time.elapsed,

            test_indices=test_indices,
            test_predictions=test_predictions,

            index_params=index_params,
            params=parameters,

            index_split=index_split
        )
        return out

    def _fit(self, X, y, **fit_params):
        """Fit each combination of parameters using k-fold cross validation

        Args:
            X (np.array): Training vector (n_samples, n_features).
            y (np.array): Target vector (n_samples).
            **fit_params : additional parameters

        Returns:
            list of dict containing:
                - estimator
                - train_score
                - test_score
                - train_time
                - inference_time
                - score_time
                - test_indices
                - test_predictions
                - index_params
                - params
                - index_split
        """
        # For sampling n_iter combinations of parameters
        params_collected = ParameterSampler(param_distributions=self.params_distribution, n_iter=self.n_iter,
                                            random_state=self.random_state)
        # This will contain a n_iter length list of parameter sampled
        params_collected = list(params_collected)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        # product is a double for-loop: for each params for each (train,test) do fit_and_score
        with parallel:
            outs = parallel(
                delayed(self._fit_and_score)(X, y, train_indices, test_indices, param, index_param, index_split, **fit_params)
                for
                (index_param, param),
                (index_split, (train_indices, test_indices))
                in product(
                    enumerate(params_collected),
                    enumerate(self.cv.split(X))))

        return outs

    def to_numpy(self, X, y):
        """Transform X and y to numpy array if they are DataFrame

        Returns:
            X and y as numpy array
        """
        if isinstance(X, (DataFrame, Series)):
            X = X.to_numpy()
        if isinstance(y, (DataFrame, Series)):
            y = y.to_numpy()
        return X, y

    def fit(self, X, y, **fit_params):
        """ Fit the estimator

        Args:
            X (np.array): Training vector (n_samples, n_features).
            y (np.array): Target vector (n_samples).
            **fit_params ():
        Returns:
            self
        """
        # Change dataframe in np.array
        X, y = self.to_numpy(X, y)

        # Set cv_results_
        results = self._fit(X, y, **fit_params)

        self.cv_results_ = pd.DataFrame.from_dict(results)

        assert len(self.cv_results_) == len(results)

        # Group result by inddex_param and compute man and std
        scores_by_index_params = self.cv_results_[['index_params', 'train_score', 'test_score']] \
            .groupby('index_params') \
            .agg(['mean', 'std'])

        # Select best index_param where the validation score is better (based on the scoring function)
        if "error" in self.scoring:
            best_index_params = scores_by_index_params['train_score', 'mean'].argmin()
        else:
            best_index_params = scores_by_index_params['train_score', 'mean'].argmax()

        # Get mean and std for train and test score
        self.best_mean_test_score_, self.best_std_test_score_ = tuple(scores_by_index_params.loc[best_index_params, 'test_score'])
        self.best_mean_train_score_, self.best_std_train_score_ = tuple(
            scores_by_index_params.loc[best_index_params, 'train_score'])

        # Takes the first element where index_params == best_index_params
        results_by_best_index_params = self.cv_results_[self.cv_results_.index_params == best_index_params].set_index(
            'index_split')
        self.best_params_ = results_by_best_index_params.loc[0, 'params']

        # Refit the estimator
        if self.refit:
            self.best_estimator_ = self._clone_base_estimator_with_params(self.best_params_).fit(X, y, **fit_params)
        else:
            if "error" in self.scoring:
                index_to_select = results_by_best_index_params['test_score'].argmin()
            else:
                index_to_select = results_by_best_index_params['test_score'].argmax()

            # select best estimator where test_score is maximum among the split tried where index_params == best_index_params
            self.best_estimator_ = results_by_best_index_params.loc[index_to_select, 'estimator']

        return self

    def predict(self, X):
        check_is_fitted(self, 'best_estimator_', 'cv_results_')
        return self.best_estimator_.predict(X)

    def score(self, X, y_true=None, **kwargs):
        check_is_fitted(self, 'best_estimator_', 'cv_results_')
        return self.best_estimator_.score(X, y_true)

    def validate_score_function(self):
        """Validate the score string given. If it is None score method of the estimator is used.

        Returns:
            Function for computing score between ground truth and predictions
        """
        # requires y_true, y_predicted
        return get_scorer(self.scoring)

    def compute_score(self, y_true, y_preds):
        """Compute the score between predictions and ground truth given a score function.

        Args:
            y_true (np.array): Gold target vector (n_samples).
            y_preds (np.array): Predicted target vector (n_samples,)

        Returns:
            score
        """
        # Get score function
        score_fn = self.validate_score_function()

        # rely on scorer function
        # alternative to this is:
        #   score_fn(estimator, X, y_true)
        # here i'm not using sign which is an attribute of any scorer function
        # in order to maximize negative scorer that should be minimized
        # due to error in plottting them after
        return score_fn._score_func(y_true, y_preds)


# Define a tuple with the attributes:
#   - operator: the operation applied by the specified mode
#   - reference: the best possible value obtained applying the operator
MODE = namedtuple('MODE', 'operator reference')


def diff(values, predictions):
    return values - predictions


def ratio(values, predictions):
    return values / predictions


class AttributeCancellationTransformer(BaseEstimator, TransformerMixin):
    """ AttributeCancellationTransformer class for discount dataset based on an input attribute.

    Args:
        base_estimator (estimator object): Estimator to use for fitting the data.
        input_attribute (str): Attribute of the dataset to use as input
        mode (str): Discounting mode to use
        params_distribution (dict): Dictionary whose key are the parameters names and values are distribution or list. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html.
        n_iter (int): Number of parameters combinations to sample for apply hyper-parameters tuning.
        cv (int): Number of splits for each Kfold.
        shuffle (bool): If True data inside each fold are shuffled
        scoring (str): Name of the score to use for the cross-validation.
        random_state (int): Seed to use for the random state.
        n_jobs (int): Number of jobs to use for the randomized grid search.

    Attributes:
        mode_ (Mode): namedtuple with attributes 'operator' and 'reference'
    """

    # It is important that diff and ratio are outside the class as normal function otherwise joblib will not save the object
    _AVAILABLE_MODES = dict(diff=MODE(diff, 0),
                            ratio=MODE(ratio, 1))
    _AVAILABLE_MODES_KEYS = _AVAILABLE_MODES.keys()

    def __init__(self, base_estimator,
                 input_attribute: str,
                 mode: str,
                 params_distribution,
                 n_iter: int,
                 cv: int,
                 shuffle: bool,
                 scoring: str,
                 random_state: int = None,
                 n_jobs: int = 1):
        # Attributes base estimator
        self.base_estimator = base_estimator
        self.params_distribution = params_distribution

        # Attributes attribute cancellation
        self.input_attribute = input_attribute

        self.mode = mode
        self.mode_ = AttributeCancellationTransformer._AVAILABLE_MODES[mode]

        # Attributes cross-validation
        self.n_iter = n_iter
        self.cv = cv
        self.shuffle = shuffle
        self.scoring = scoring
        self.n_jobs = n_jobs

        self.random_state = random_state

        self.validate_init_params()

        self.attribute_information_ = None

    def validate_init_params(self):
        """ Checks that base_estimator and mode are correct. """
        self.base_estimator = validate_estimator(self.base_estimator, self.random_state)

        assert isinstance(self.input_attribute, str), 'input_attribute must be a column name (string)'

        assert self.mode in list(AttributeCancellationTransformer._AVAILABLE_MODES_KEYS), \
            f"mode must be: {list(AttributeCancellationTransformer._AVAILABLE_MODES_KEYS)}"

    def validate_input(self, X_df):
        """ Check that the input is correct """
        assert isinstance(X_df, DataFrame), "X must be a DataFrame."
        assert self.input_attribute in X_df.columns, "input_attribute must be the column's name of X_df"

    def fit(self, X_df, y=None, **fit_params):
        self.validate_input(X_df)

        # Get the input
        X = X_df.loc[:, self.input_attribute].values.reshape(-1, 1)
        X_df: DataFrame = X_df.drop(columns=[self.input_attribute])

        if IS_TQDM_INSTALLED:
            max_columns = len(X_df.columns)
            iterator = tqdm(X_df.iteritems(), total=max_columns, unit='attribute')
        else:
            iterator = X_df.iteritems()

        # use always 1 job due to no gains in performance
        parallel_n_jobs = 1
        info_per_attribute = Parallel(n_jobs=parallel_n_jobs)(
            delayed(self._find_best_estimator)(X, target, attribute, **fit_params)
            for attribute, target in iterator)

        # Collect results
        self.attribute_information_ = dict()
        for attribute, best_estimator, cv_results in info_per_attribute:
            self.attribute_information_[attribute] = Bunch(estimator=best_estimator, cv_results=cv_results)

        return self

    def transform(self, X_df, y=None):
        # Check X_df is correct and transformer is fitted
        self.validate_input(X_df)

        check_is_fitted(self, 'attribute_information_')

        # remove input and target
        input_column = X_df.loc[:, self.input_attribute]

        data = X_df.drop(columns=[self.input_attribute])

        fn_to_apply = partial(self._apply_discount, input=input_column)

        return pd.concat([input_column, data.apply(fn_to_apply, axis=_OPERATION_COLUMN_WISE)],
                         axis=_OPERATION_ROW_WISE)

    def _find_best_estimator(self, X, y, attribute, **fit_params):
        # Get params for cross validation
        cv = CrossValidation(base_estimator=clone(self.base_estimator),
                             params_distribution=self.params_distribution,
                             n_iter=self.n_iter,
                             cv=self.cv,
                             shuffle=self.shuffle,
                             pre_dispatch="n_jobs",
                             n_jobs=self.n_jobs,
                             scoring=self.scoring,
                             random_state=self.random_state) \
            .fit(X, y, **fit_params)
        return attribute, cv.best_estimator_, cv.cv_results_

    def _apply_discount(self, target, input):
        operator = AttributeCancellationTransformer._AVAILABLE_MODES[self.mode].operator
        input_values = input.values.reshape(-1, 1)
        predictions = self.attribute_information_[target.name].estimator.predict(input_values)
        # operator(pandas.Series, numpy.array) -> pandas.Series
        return operator(input, predictions)
