adaptivesplit
==========================
Adaptive training-validation split during prospective data acquisition to improve the development and external validation of predictive models.

Scikit-Learn example usage:
    
    from adaptivesplit.sklearn_interface.split import AdaptiveSplit
    
    X = "your predictors here"
    y = "your target here"
    
    model = "an sklearn estimator"
  
    adsplit = AdaptiveSplit(total_sample_size=len(y), plotting=True)
    res, fig = adsplit(X, y, model, fast_mode=True, predict=False, random_state=42)
    stop = res.estimated_stop

This prints out the results and plots the learning and power curves.
