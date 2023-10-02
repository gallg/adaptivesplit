from ..base.utils import *
from sklearn.metrics import get_scorer as sklearn_get_scorer
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
import numpy as np


def get_sklearn_scorer(scoring):
    """Provides a scikit-learn scoring function given an input string.

    Args:
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

            If None, the estimator’s score method is used. Defaults to "scoring" as specified in the configuration file.

    Returns:
        score_func (callable):
            Scikit-Learn scoring function.
    """
    scoring = sklearn_get_scorer(scoring)

    def score_fun(x, y, **kwargs):
        return scoring._sign * scoring._score_func(x, y) 
        # _score_func had a **kwargs argument that amounted to {random_seed:None}
        # it crashes since the function does not accept a random_seed argument
        # see also power.py;

    score_fun.__name__ = 'sklearn_' + scoring._score_func.__name__
    return score_fun


def statfun_as_callable(stat_fun):
    """Returns a statistical function.

    Args:
        stat_fun (str, callable): 
            If this is a str, use sklearn.metrics.get_sklearn_scorer to make
            stat_fun a callable.

    Returns:
        stat_fun (callable): 
            Statistical function.
    """
    if isinstance(stat_fun, str):
        return get_sklearn_scorer(stat_fun)
    else:
        return stat_fun


def calculate_ci(X, ci='95%'):
    """Calculate confidence intervals.

    Args:
        X (list, np.ndarray, pd.Series): 
            1D array of shape (n_samples,).
        ci (str, optional): 
            Confidence level to Return. Defaults to '95%'.
            90%, 95%, 98%, 99% are possible inputs. 
            
    Returns:
        ci_lower: 
            Confidence intervals lower bound.
        ci_upper:
            Confidence intervals upper bound.
    """

    if ci == '90%':
        Z = 1.64
    elif ci == '95%':
        Z = 1.96
    elif ci == '98%':
        Z = 2.33
    elif ci == '99%':
        Z = 2.58

    moe = Z*(np.std(X)/np.sqrt(len(X)))
    ci_lower = np.mean(X) - moe
    ci_upper = np.mean(X) + moe
    
    return ci_lower, ci_upper
