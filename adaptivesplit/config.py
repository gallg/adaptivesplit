import os
import configparser
import warnings
import numpy as np
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class CustomConfigParser(configparser.ConfigParser):
    """Custom ConfigParser implementing interpolation. Inherits from configparser.ConfigParser. 
    Used to read AdaptiveSplit's configuration file which contains the default values for the
    AdaptiveSplit main class and its 'stopping_rule' method. Parameters for the main class are
    specified in the 'adaptivesplit.sklearn_interface.split' section of the documentation, while
    the parameters for the stopping rule are described here.
    
    Args:
        min_training_sample_size (int):
            If not 0, the stopping rule considers necessary reaching a number of samples equal to
            this paramater before finding a stop point. If 'min_score' is not equal to -inf then
            reaching the minimum training sample size and the minimum score acts as precondition for
            all the other rules to be evaluated. Defaults to the corresponding value in the configuration
            file. 
        target_power (float):
            Target power to reach before a stopping point can be found. If 0, power rule is not evaluated.
            Defaults to the corresponding value in the configuration file. 
        alpha (float):
            Target p-val to reach before the slope of the power curve is calculated. Defaults to the 
            corresponding value in the configuration file. 
        min_score (int or float):
            Minimum score to reach before finding a stopping point. See also the 'min_training_sample_size'
            argument for more information. Defaults to the corresponding value in the configuration file. 
        min_relevant_score (int or float):
            If not 0, learning curve rule is evaluated and a stopping point is found when the scores stabilize
            (useful in case of power curve plateaus). Defaults to the corresponding value in the configuration
            file. 
        min_validation_sample_size (int):
            If not 0, specifies the minimum size of the validation sample, which coincides with the last possible
            stopping point. Defaults to the corresponding value in the configuration file. 
    """
    
    def getlist(self, section, option):
        return json.loads(self.get(section, option))

settings = CustomConfigParser(allow_no_value=True)
settings.read(os.path.join(ROOT_DIR, 'settings.conf'))

# Stopping rule settings
_min_training_sample_size_ = settings.getint('STOPPING_RULE', 'min_training_sample_size')
_target_power_ = settings.getfloat('STOPPING_RULE', 'target_power')
_alpha_ = settings.getfloat('STOPPING_RULE', 'alpha')
_min_relevant_score_ = settings.getfloat('STOPPING_RULE', 'min_relevant_score')
_min_validation_sample_size_ = settings.getfloat('STOPPING_RULE', 'min_validation_sample_size')
_window_size_ = settings.getint('STOPPING_RULE', 'window_size')
_step_ = settings.getint('STOPPING_RULE', 'step')


try:
    _min_score_ = settings.getfloat('STOPPING_RULE', 'min_score')
except:
    warnings.warn("min_score expected as float, is set to -np.inf")
    _min_score_ = -np.inf

# ------------------------------------------------
# AdaptiveSplit settings
_cv_ = settings.getint('AdaptiveSplit', 'cv')
_bootstrap_samples_ = settings.getint('AdaptiveSplit', 'bootstrap_samples')
_power_bootstrap_samples_ = settings.getint('AdaptiveSplit', 'power_bootstrap_samples')
_n_jobs_ = settings.getint('AdaptiveSplit', 'n_jobs')
_scoring_ = settings['AdaptiveSplit']['scoring']
_total_sample_size_ = settings.getint('AdaptiveSplit', 'total_sample_size')
_stratify_ = settings.getint('AdaptiveSplit', 'stratify')
_fast_mode_ = settings.getboolean('AdaptiveSplit', 'fast_mode')
_sample_size_multiplier_ = settings.getfloat('AdaptiveSplit', 'sample_size_multiplier')

if _stratify_ == 0:
    _stratify_ = None
