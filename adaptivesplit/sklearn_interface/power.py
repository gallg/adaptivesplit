# extends adaptivesplit.base.power with scikit-learn functionality
import collections
import pandas as pd
from ..base.learning_curve import LearningCurve
from sklearn.model_selection import cross_val_predict
from ..base.power import *
from ..base.power import _get_sample_sizes
from .resampling import Resample
from .utils import statfun_as_callable

# shuffleCV predict x_y
# y, y_pred
# subsample, y, y_pred
# power for subsamples
# score for subsamples
#
#
# power curve predicted
# score confidence interval curve predicted

PredictedScoreAndPower = collections.namedtuple('Predicted', ['score', 'power'])
"""Returned by the "predict_power_curve" function."""


def predict_power_curve(estimator, X, y, power_estimator,
                        total_sample_size, stratify=None, sample_sizes=None, step=None,
                        cv=5,
                        num_samples=100,
                        scoring=None, verbose=True,
                        n_jobs=None,
                        random_state=None,  # todo: implement it!
                        **kwargs):
    """If total_sample_size > len(y) predicts the power curve trend to show what happens
       when the sample size is higher.

    Args:
        estimator (estimator object): 
            This is assumed to implement the scikit-learn estimator interface.
        X (numpy.ndarray or pandas.DataFrame): 
            array-like of shape (n_samples, n_features). The data to fit as in scikit-learn. Can be a numpy array or 
            pandas DataFrame.
        y (numpy.ndarray or pandas.Series): 
            array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target variable to try to predict in the case of supervised learning, as in scikit-learn.
        power_estimator (callable): 
            Must be a power_estimator function, see the 'create_power_estimator*' factory functions.
        total_sample_size (int): 
            The total number of samples in the data given as input.
        stratify (int): 
            For classification tasks. If not None, use stratified sampling to account for class labels imbalance. 
            Defaults to None.
        sample_sizes (int or list of int): 
            Sample sizes to calculate the power curve. Defaults to None.
        step (int): 
            Step size between sample sizes. A value of 1 is recommended. Defaults to None.
        cv (int, cross-validation generator or an interable): 
            Determines the cross-validation splitting strategy, as in scikit-learn. Possible inputs for cv are:
            
            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a (Stratified)KFold,
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.
            
            For int/None inputs, if the estimator is a classifier and y is either binary or multiclass,
            StratifiedKFold is used. In all other cases, K-Fold is used. These splitters are instantiated
            with shuffle=False so the splits will be the same across calls. Defaults to 5.
        num_samples (int): 
            Number of iterations to shuffle data before determining subsamples. The first iteration 
            (index 0) is ALWAYS unshuffled (num_samples=1 implies no resampling at all, default). 
            Defaults to 100.
        scoring (str, callable, list, tuple or dict): 
            Scikit-learn-like score to evaluate the performance of the cross-validated model on the test set.
            If scoring represents a single score, one can use:

            - a single string (see The scoring parameter: defining model evaluation rules);
            - a callable (see Defining your scoring strategy from metric functions) that returns a single value.

            If scoring represents multiple scores, one can use:

            - a list or tuple of unique strings;
            - a callable returning a dictionary where the keys are the metric names and the values are the
              metric scores;
            - a dictionary with metric names as keys and callables a values.

            If None, the estimator’s score method is used. Defaults to None.
        verbose (bool): 
            Prints progress. Defaults to True.
        n_jobs (int): 
            Number of jobs to run in parallel. Defaults to None. Training the estimator and computing the score are 
            parallelized over the cross-validation splits. None means 1 unless in a joblib.parallel_backend context.
            -1 means using all processors.
        random_state (int): 
            Controls the randomness of the bootstrapping of the samples used when building sub-samples (if shuffle!=-1). 
            Defaults to None. Currently NOT implemented.

    Returns:
        PredictedScoreAndPower (tuple): 
            Contains the predicted score and power (in this order). 
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = np.squeeze(y.to_numpy())  # squeezing avoids errors with some datasets;
    else:
        y = np.squeeze(y)

    sample_sizes = _get_sample_sizes(sample_sizes, step, len(y), total_sample_size)
    sample_sizes_power = total_sample_size - sample_sizes

    power_estimator.total_sample_size = None

    if scoring is None:
        # scoring = estimator.score  # todo does this work?
        pass  # no, it does not work, requires fitted estimator. pass for now;
    else:
        scoring = statfun_as_callable(scoring)

    def stat_fun_score_and_power(_x, _y, **kwargs):
        if len(_y) == 0:
            score = np.nan
            power = 0
        else:
            score = scoring(_x, _y, **kwargs)
            power = power_estimator.estimate(_x, _y, sample_size=len(_y),
                                             verbose=False)[0]
        return PredictedScoreAndPower(score, power)

    def stat_fun_aggregate_samplesizes(_x, _y, **kwargs):
        # data has been shuffled
        # do cv prediction forsample_sizes whole sample
        pred_y = cross_val_predict(estimator, _x, _y,
                                   cv=cv)
        # cross_val_predict had a **kwargs argument that amounted to {random_seed:None}
        # it crashes since the function doesn't accept a random_seed argument
        # see also utils.py;

        # subsample score
        subsampler = Resample(stat_fun=stat_fun_score_and_power,
                              sample_size=sample_sizes_power,
                              num_samples=1,  # we use the outer scope
                              n_jobs=1,
                              verbose=False,
                              replacement=True)
        scores_and_powers = subsampler.bootstrap(pred_y, _y)
        return scores_and_powers  # per subsample

    shuffler = Resample(stat_fun=stat_fun_aggregate_samplesizes,
                        sample_size=len(y),
                        num_samples=num_samples,
                        n_jobs=n_jobs,
                        verbose=verbose,
                        replacement=False,
                        message="Predict Power Curve"
                        )

    results = shuffler.subsample(X, y, stratify=stratify)

    results = results.squeeze().transpose([2, 0, 1])  # (type, bootstrap_iters, samples)

    pred_score = LearningCurve(data=results[0],
                               ns=sample_sizes,
                               scoring=scoring.__name__,
                               curve_type="predicted score",
                               description="predicted score",
                               )

    pred_power = LearningCurve(data=results[1],
                               ns=sample_sizes,
                               scoring=str(power_estimator),
                               curve_type="predicted power",
                               description="predicted power",
                               )

    return PredictedScoreAndPower(pred_score, pred_power)
