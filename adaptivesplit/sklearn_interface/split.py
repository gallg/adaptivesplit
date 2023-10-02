import collections
import pandas as pd
from ..base.split import *
from .power import PowerEstimatorBootstrap, create_power_stat_fun, predict_power_curve
from .learning_curve import calculate_learning_curve
from sklearn.base import is_classifier
from .utils import calculate_ci, check_cv, statfun_as_callable
import adaptivesplit.config as config
from sklearn import linear_model
from regressors import stats
from pygam import LinearGAM

AdaptiveSplitResults = collections.namedtuple('AdaptiveSplitResults', ['stop', 'predicted',
                                                                       'estimated_stop',
                                                                       'current_sample_size',
                                                                       'score_if_stop',
                                                                       'score_if_stop_now_ci',
                                                                       'power_if_stop_now',
                                                                       'power_if_stop_now_ci',
                                                                       'score_predicted',
                                                                       'score_predicted_ci',
                                                                       'power_predicted',
                                                                       'power_predicted_ci'])


def _calc_slope(series, alpha=0.05):

    if isinstance(series, pd.Series):
        series = np.array(series)

    Y = series.reshape(-1, 1)
    X = np.arange(series.shape[0]).reshape(-1, 1)

    model = linear_model.LinearRegression().fit(X, Y)
    p_vals = stats.coef_pval(model, X, Y)

    if max(p_vals) < alpha:
        return model.coef_[0][0]
    else:
        return np.NAN


def _pred_score(series, x_test, alpha=0.05):

    if isinstance(series, pd.Series):
        series = np.array(series)
    if isinstance(x_test, pd.Series):
        series = np.array(x_test)

    Y = series.reshape(-1, 1)
    X = np.arange(series.shape[0]).reshape(-1, 1)

    model = linear_model.LinearRegression().fit(X, Y)
    p_vals = stats.coef_pval(model, X, Y)

    if x_test is None:
        x_test = X[-1]

    if max(p_vals) < alpha and model.coef_[0][0] > 0:
        return model.predict(np.array(x_test).reshape(-1, 1))
    else:
        return np.NAN


class AdaptiveSplit:
    """Run the AdaptiveSplit model. This evaluates performance on multiple splits of the data by calculating
    the learning and power curves using bootstrap. The model works for both regression and classification
    tasks, depending on the scikit-learn estimator and type of score metric provided.

    If the total sample size provided to this class is higher than len(Y), the algorithm will predict the 
    learning and power curves for the additional samples. This is useful to check if a higher sample size is
    able to enhance model prediction.

    Args:
        total_sample_size (int): 
            The total length of the data given as input. Defaults to "total_sample_size" as specified 
            in the configuration file.
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
        cv (int, cross-validation generator or an iterable):
            Determines the cross-validation splitting strategy, as in scikit-learn. Possible inputs for cv are:
            
            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a (Stratified)KFold,
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.
            
            For int/None inputs, if the estimator is a classifier and y is either binary or multiclass,
            StratifiedKFold is used. In all other cases, K-Fold is used. These splitters are instantiated
            with shuffle=False so the splits will be the same across calls. Defaults to "cv" as specified 
            in the configuration file.
        step (int): 
           Step size between sample sizes. A value of 1 is recommended. Defaults to "step" as specified in
           the configuration file.
        bootstrap_samples (int): 
            Number of samples generated during bootstrapping. Defaults to "bootstrap_samples" as
            specified in the configuration file.
        power_bootstrap_samples (int): 
            Number of iteration during which samples are bootstrapped to calculate power. Defaults to
            "power_bootstrap_samples" as specified in the configuration file.
        window_size (int): 
            Size of the rolling window used to calculate the slope of the power curve. 
            if fast_mode in the fit method is equal to true it is also used to calculate
            reduces sample sizes to use into fast mode. If None, defaults to "window_size" as
            specified in the configuration file.
        verbose (bool): 
            Prints progress. Defaults to True.
        plotting (bool): 
            Whether or not to plot the learning and the power curves after calculations. Defaults to True.
        ci (str):
            Intervals confidence used when plotting the learning and power curves. Defaults to 95%.
        n_jobs (int):
            Number of jobs to run in parallel. Defaults to "n_jobs" as specified in the configuration file.
            Training the estimator and computing the score are parallelized over the cross-validation splits.
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

    Returns:
        AdaptiveSplitResults (namedtuple):
            Contains the results from the AdaptiveSplit algorithm, i.e. estimated stopping point, scores and power.
        Figure (matplotlib.figure.Figure):
            Plot illustrating the learning and power curves.

    """
    def __init__(self, total_sample_size=config._total_sample_size_, scoring=config._scoring_, cv=config._cv_,
                 step=config._step_, bootstrap_samples=config._bootstrap_samples_, 
                 power_bootstrap_samples=None, window_size=None, verbose=True, 
                 plotting=True, ci='95%', n_jobs=config._n_jobs_):
       
        self.total_sample_size = total_sample_size
        self.scoring = scoring
        self.cv = cv
        self.step = step
        self.bootstrap_samples = bootstrap_samples
        self.verbose = verbose
        self.plotting = plotting
        self.n_jobs = n_jobs
        self.reason = []
        self.window_size = window_size
        self.ci = ci

        if not window_size:
            self.window_size = config._window_size_
        if not power_bootstrap_samples:
            self.power_bootstrap_samples = config._power_bootstrap_samples_
        else:
            self.power_bootstrap_samples = power_bootstrap_samples
            
    def fit(self, X, Y, estimator, stratify=config._stratify_, fast_mode=config._fast_mode_, 
            sample_size_multiplier=config._sample_size_multiplier_, predict=True, random_state=None):
        """Fit the AdaptiveSplit model.

        Args:
            X (numpy.ndarray or pandas.DataFrame):
                array-like of shape (n_samples, n_features). The data to fit as in scikit-learn. Can be a numpy array or 
                pandas DataFrame.
            Y (numpy.ndarray or pandas.Series):
                array-like of shape (n_samples,) or (n_samples, n_outputs).
                The target variable to try to predict in the case of supervised learning, as in scikit-learn.
            estimator (estimator object): 
                Estimator object. A object of that type is instantiated for each grid point.
                This is assumed to implement the scikit-learn estimator interface.
                Either estimator needs to provide a score function, or scoring must be passed.
                If it is e.g. a GridSearchCV then nested cv is performed (recommended).
            stratify (_type_, optional): 
                For classification tasks. If not None, use stratified sampling to account for class labels imbalance.
                Defaults to "stratify" as specified in the configuration file.
            fast_mode (bool): 
                If True the algorithm is evaluated on reduced sample sizes to reduce runtime. Defaults to "fast_mode"
                as specified in the configuration file.
            sample_size_multiplier (float): 
                Multiplier value to make sure the algorithm starts with adequate sample sizes.
                (Recommended value is 0.2). Defaults to "sample_size_multiplier" as specified in the configuration file.
            predict (bool, optional): 
                If True, try to predict the learning and power curve for additional samples.
                If total_sample_size == len(Y) it automatically turns to False. Defaults to True.
            random_state (int, optional): 
                Controls the randomness of the bootstrapping of the samples used when building sub-samples 
                (if shuffle!=-1). Defaults to None.
        """

        self.X = X
        self.Y = Y
        self.estimator = estimator
        self.fast_mode = fast_mode
        self.predict = predict
        self.random_state = random_state

        if self.total_sample_size < len(Y):
            raise AttributeError(
                "total_sample_size must be greater than or equal to the actual sample size: " + str(len(Y)))
        
        self.power_estimator = PowerEstimatorBootstrap(
            power_stat_fun=create_power_stat_fun('permtest', statfun_as_callable(self.scoring), num_perm=100),
            bootstrap_samples=self.power_bootstrap_samples,  # it will be bootstrapped in the outer loop anyway
            total_sample_size=self.total_sample_size,
            alpha=config._alpha_,
            stratify=stratify,
            verbose=False
        )

        self.cv = check_cv(self.cv, self.Y, classifier=is_classifier(self.estimator))
        window_size = self.window_size

        # Calculate the starting point here;
        if self.fast_mode:
            sample_sizes = np.arange(start=len(self.Y), stop=len(self.Y) - window_size*2, step=-self.step)[::-1]
            # window_size is multiplied for 2 in fast mode to have enough samples for the GAM;
            # Adapt for classifier?
        else:
            if is_classifier(estimator):
                
                # search for the label with the least members
                labels, counts = np.unique(self.Y, return_counts=True)
                ratio_of_smallest_class = np.min(counts)/len(self.Y)

                # make sure to start with an adequate starting_sample_size
                starting_sample_sizes = np.arange(start=len(self.Y), stop=self.cv.get_n_splits(), step=-self.step)[::-1]
                sample_sizes = starting_sample_sizes[ratio_of_smallest_class*starting_sample_sizes > self.cv.n_splits]
                
                # check if sample_sizes is calculated properly 
                # (sample_sizes might be an empty array if ratio_of_smallest_class is too small)
                if len(sample_sizes) == 0:
                    sample_sizes = starting_sample_sizes[sample_size_multiplier*starting_sample_sizes > self.cv.n_splits]

            else:
                starting_sample_sizes = np.arange(start=len(self.Y), stop=self.cv.get_n_splits(), step=-self.step)[::-1]
                sample_sizes = starting_sample_sizes[sample_size_multiplier*starting_sample_sizes > self.cv.n_splits]

        self.lc_train, self.lc_test, self.lc_dummy, self.lc_power = calculate_learning_curve(
            estimator=self.estimator,
            X=self.X,
            y=self.Y,
            cv=self.cv,
            stratify=stratify,
            power_estimator=self.power_estimator,
            scoring=self.scoring,
            sample_sizes=sample_sizes,
            num_samples=self.bootstrap_samples,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        if self.total_sample_size == len(self.Y):
            predict = False
        
        if not predict:
            self.pred_power = None
            self.pred_score = None
            self.score = self.lc_test.stat().df['mid']
            self.score_upper = self.lc_test.stat().df['upper']
            self.score_lower = self.lc_test.stat().df['lower']
            self.power = self.lc_power.stat().df['mid']
            self.power_upper = self.lc_power.stat().df['upper']
            self.power_lower = self.lc_power.stat().df['lower']
        
        elif predict:
            pred = predict_power_curve(estimator=self.estimator,
                                       X=self.X,
                                       y=self.Y,
                                       power_estimator=self.power_estimator,
                                       total_sample_size=self.total_sample_size,
                                       step=self.step,
                                       cv=self.cv,
                                       num_samples=100,       # Bootstrap
                                       scoring=self.scoring,  # None does not work
                                       verbose=True,
                                       n_jobs=self.n_jobs,
                                       random_state=None
                                       )
            self.score = pd.concat([self.lc_test.stat().df['mid'], pred.score.stat().df.iloc[1:, 0]])
            self.score_upper = pd.concat([self.lc_test.stat().df['upper'], pred.score.stat().df.iloc[1:, 1]])
            self.score_lower = pd.concat([self.lc_test.stat().df['lower'], pred.score.stat().df.iloc[1:, 2]])
            self.power = pd.concat([self.lc_power.stat().df['mid'], pred.power.stat().df.iloc[1:, 0]])
            self.power_upper = pd.concat([self.lc_power.stat().df['upper'], pred.power.stat().df.iloc[1:, 1]])
            self.power_lower = pd.concat([self.lc_power.stat().df['lower'], pred.power.stat().df.iloc[1:, 2]])
            self.pred_power = pred.power
            self.pred_score = pred.score

    def stopping_rule(self):
        min_training_sample_size = config._min_training_sample_size_        # 0 means this rule is switched off
        min_validation_sample_size = config._min_validation_sample_size_    # 0 means there is no max stopping point
        target_power = config._target_power_                                # 0 mean this rule is switched off
        min_score = config._min_score_                                      # -np.inf means this rule is switched off
        
        window_size = self.window_size
            
        min_relevant_score = config._min_relevant_score_

        if not self.predict:
            stop = self.total_sample_size
        else:
            stop = self.score.index.max()

        # Generate a smoother power curve using GAM; Todo: we should also smooth the prediction, if present;
        power_gam = []

        for sample_i, actual_sample_size in enumerate(self.power.index):

            if sample_i == 0:
                gam_X = np.array([self.power.index[sample_i]])[:, np.newaxis]
                gam_y = np.array([self.power.iloc[sample_i]])
            else:
                gam_X = np.array(self.power.index)[:sample_i][:, np.newaxis]
                gam_y = np.array(self.power)[:sample_i]

            gam = LinearGAM().gridsearch(gam_X, gam_y, n_splines=np.arange(1, 6))
            X_grid = gam.generate_X_grid(term=0, n=len(gam_X))
            power_gam.append(gam.predict(X_grid))
            
        # Prepare the power curve for rule evaluation and plotting;
        mean_power_curve = pd.DataFrame(power_gam, columns=self.power.index[1:]).mean(axis=0)
        ci_lower, ci_upper = calculate_ci(mean_power_curve, ci=self.ci)
        
        self.power = mean_power_curve
        self.power_lower = mean_power_curve - ci_lower
        self.power_upper = mean_power_curve + ci_upper

        power_slope = mean_power_curve.rolling(window_size).apply(_calc_slope, raw=False)   # NaN if NOT significant
        pred_final_score = self.score.rolling(window_size).apply(_pred_score, raw=False, args=(self.total_sample_size,))
        pred_actual_score = self.score.rolling(window_size).apply(_pred_score, raw=False, args=(None,))

        # Evaluate the rule;
        for actual_sample_size in mean_power_curve.index:
            self.reason = []

            # max training size exceeded:
            if not self.fast_mode:
                if actual_sample_size >= self.total_sample_size - min_validation_sample_size:
                    stop = actual_sample_size
                    self.reason.append('max sample size reached')
            else:
                if actual_sample_size >= self.total_sample_size:
                    stop = actual_sample_size
                    self.reason.append('max sample size reached')

            # preconditions for all other rules: min training size and min score must be exceeded
            if actual_sample_size > min_training_sample_size \
                    and self.score.loc[actual_sample_size] > min_score:

                # power rule: power is already decreasing and we pass by the target power
                if power_slope[actual_sample_size] < 0 and mean_power_curve.loc[actual_sample_size] <= target_power:
                    stop = actual_sample_size
                    self.reason.append('power rule')

                # score rule: optimistic (linear) extrapolation of the power curve predicts very little gain,
                # i.e. power curve plateaus
                if (pred_final_score[actual_sample_size] - pred_actual_score[actual_sample_size]) < min_relevant_score:
                    stop = actual_sample_size
                    self.reason.append('score rule')

            if stop == actual_sample_size:
                if len(self.reason) > 1:
                    self.reason = ', '.join(self.reason)
                break
            elif stop == self.total_sample_size:
                self.reason.append('No stopping point found')

        # Set up plotting and return AdaptiveSplit results;
        if self.plotting:
            fig = plot(learning_curve=self.lc_test, learning_curve_predicted=self.pred_score,
                       power_curve=self.power, power_curve_lower=self.power_lower, power_curve_upper=self.power_upper,
                       power_curve_predicted=self.pred_power, training_curve=self.lc_train, dummy_curve=self.lc_dummy,
                       stop=stop, reason=self.reason, ci=self.ci)
        else:
            fig = plt.figure() 

        if stop > len(self.Y):
            is_stop_predicted = True
        else:
            is_stop_predicted = False

        current = len(self.Y)

        if not self.predict:
            return AdaptiveSplitResults(stop <= current,
                                        is_stop_predicted,
                                        stop,
                                        current,

                                        self.score[current],
                                        (self.score_lower[current], self.score_upper[current]),
                                        self.power[current],
                                        (self.power_lower[current], self.power_upper[current]),
                                        
                                        None,
                                        (None, None),
                                        None,
                                        (None, None),
                                        ), fig
        else: 
            return AdaptiveSplitResults(stop <= current,
                                        is_stop_predicted,
                                        stop,
                                        current,

                                        self.score[current],
                                        (self.score_lower[current], self.score_upper[current]),
                                        self.power[current],
                                        (self.power_lower[current], self.power_upper[current]),
                                        
                                        self.score[stop],
                                        (self.score_lower[stop], self.score_upper[stop]),
                                        self.power[stop],
                                        (self.power_lower[stop], self.power_upper[stop]),
                                        ), fig

    def __call__(self, data, target, estimator, stratify=None, fast_mode=False, predict=False, random_state=None):

        self.fit(X=data, Y=target, estimator=estimator, stratify=stratify, fast_mode=fast_mode, predict=predict, 
                 random_state=random_state)
        res, fig = self.stopping_rule()
        return res, fig
        