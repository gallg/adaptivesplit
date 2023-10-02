# extends basse.learning_curve with scikit-learn functionality
from ..base.learning_curve import *
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.base import is_classifier, is_regressor
from .utils import check_cv, check_scoring
from .resampling import SubSampleCV

"""
def lc_keymaker(estimator, X, y, ns, cv=5, cv_stat=np.mean, dummy_estimator=None,
                shuffle=-1, replacement=False, scoring=None, verbose=True, n_jobs=None, random_state=None,
                *args, **kwargs):
    # resolve nans:
    if dummy_estimator is None:
        dummy_estimator = 'default'
    if scoring is None:
        scoring = 'default'
    if n_jobs is None:
        n_jobs = 'default'
    if random_state is None:
        random_state = np.random.normal()

    return str(estimator), str(X), str(y), str(ns), str(cv), str(
        cv_stat), dummy_estimator, shuffle, replacement, scoring, verbose, n_jobs, random_state
"""


# factory function for sklearn
# @cached(max_size=64, custom_key_maker=lc_keymaker)
def calculate_learning_curve(estimator, X, y, sample_sizes, stratify=None, cv=5, cv_stat=np.mean, dummy_estimator=None,
                             num_samples=1,
                             power_estimator=None,
                             scoring=None, verbose=True,
                             n_jobs=None,
                             random_state=None,
                             *args, **kwargs):
    """Calculate learning curve on training and test data. Also generates a learning
       curve for baseline performance using dummy estimators.

    Args:
        estimator (estimator object): 
            Estimator object. A object of that type is instantiated for each grid point.
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a score function, or scoring must be passed.
            If it is e.g. a GridSearchCV then nested cv is performed (recommended).
        X (numpy.ndarray or pandas.DataFrame):
            array-like of shape (n_samples, n_features). The data to fit as in scikit-learn.
            Can be a numpy array or pandas DataFrame.
        y (numpy.ndarray or pandas.Series):
            array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target variable to try to predict in the case of supervised learning,
            as in scikit-learn.
        sample_sizes (int or list of int):
            sample sizes to calculate the learning curve.
        stratify (int):
            For classification tasks. If not None, use stratified sampling to account for
            class labels imbalance. Defaults to None.
        cv (int, cross-validation generator or an iterable):
            Determines the cross-validation splitting strategy, as in scikit-learn.
            Possible inputs for cv are:
            
            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a (Stratified)KFold,
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.
            
            For int/None inputs, if the estimator is a classifier and y is either binary or
            multiclass, StratifiedKFold is used. In all other cases, K-Fold is used. These
            splitters are instantiated with shuffle=False so  the splits will be the same
            across calls. Defaults to 5.
        cv_stat (callable):
            Function for aggregating cross-validation-wise scores. Defaults to numpy.mean.
        dummy_estimator (estimator object):
            A scikit-learn-like dummy estimator to evaluate baseline performance.
            If None, either DummyClassifier() or DummyRegressor() are used, based on 'estimator's type.
        num_samples (int):
            Number of iterations to shuffle data before determining subsamples.
            The first iteration (index 0) is ALWAYS unshuffled (num_samples=1 implies no resampling at all, default).
        power_estimator (callable):
            Callable must be a power_estimator function, see the 'create_power_estimator*' factory functions.
            If None, power curve is not calculated. Defaults to None.
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
            If not False, prints progress. Defaults to True.
        n_jobs (int):
            Number of jobs to run in parallel. Defaults to None.
            Training the estimator and computing the score are parallelized over the cross-validation splits.
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
        random_state (int):
            Controls the randomness of the bootstrapping of the samples used when building sub-samples 
            (if shuffle!=-1). Defaults to None.
        *args: 
            Extra parameters passed to sklearn.model_selection.cross_validate.
        **kwargs:
            Extra keyword parameters passed to sklearn.model_selection.cross_validate.

    Returns:
        lc_train (adaptivesplit.base.learning_curve.LearningCurve object): 
            Learning curve calculated on training data.
        lc_test (adaptivesplit.base.learning_curve.LearningCurve object):
            Learning curve calculated on test data.
        lc_dummy (adaptivesplit.base.learning_curve.LearningCurve object):
            Learning curve calculated using the dummy estimator. It estimates baseline learning performance.
    """

    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = np.squeeze(y.to_numpy())  # squeezing avoids errors with some datasets;
    else:
        y = np.squeeze(y)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    if isinstance(sample_sizes, (int, float)):
        inc = (len(y) - cv.get_n_splits()) / sample_sizes
        sample_sizes = np.arange(start=cv.get_n_splits(), stop=len(y)+inc, step=inc)

    if scoring is None:
        scoring = check_scoring(estimator)

    if dummy_estimator is None:
        if is_classifier(estimator):
            dummy_estimator = DummyClassifier()  # strategy='stratified'?
        elif is_regressor(estimator):
            dummy_estimator = DummyRegressor()
        else:
            raise RuntimeError("Estimator can only be classifier or regressor.")

    subsampler = SubSampleCV(estimator=estimator,
                             dummy_estimator=dummy_estimator,
                             sample_size=sample_sizes,
                             num_samples=num_samples,
                             cv=cv,
                             cv_stat=cv_stat,
                             power_estimator=power_estimator,
                             scoring=scoring,
                             verbose=verbose,
                             n_jobs=n_jobs
                             )
    stats = subsampler.subsample(X, y, stratify=stratify, random_seed=random_state)

    # return the stuff
    lc_train = LearningCurve(data=stats[0, :, :],
                             ns=sample_sizes,
                             scoring=scoring,
                             description={
                                 "shuffles": num_samples},
                             curve_type="train"
                             )

    lc_test = LearningCurve(data=stats[1, :, :],
                            ns=sample_sizes,
                            scoring=scoring,
                            description={
                                "shuffles": num_samples},
                            curve_type="test"
                            )

    lc_dummy = LearningCurve(data=stats[2, :, :],
                             ns=sample_sizes,
                             scoring=scoring,
                             description={
                                 "shuffles": num_samples},
                             curve_type="dummy"
                             )

    if power_estimator is not None:
        lc_power = LearningCurve(data=stats[3, :, :],
                                 ns=sample_sizes,
                                 scoring=scoring,
                                 description={
                                     "shuffles": num_samples},
                                 curve_type="power"
                                 )
        return lc_train, lc_test, lc_dummy, lc_power

    return lc_train, lc_test, lc_dummy
