---
title: "External validation of machine learning models: registered models and adaptive sample splitting"
subject: Preprint
short_title: AdaptiveSplit Preprint

exports:
  - format: pdf
    template: springer
    output: exports/adaptivesplit_manuscript.pdf    
  - format: docx
    output: exports/adaptivesplit_manuscript.docx

#bibliography:
#  - bibliography.bib
---

+++ {"part": "key-points"}
**Key Points:**
- todo
+++

+++ {"part": "abstract"}
Complex and intricate models, as well as extensive data pre-processing pipelines, are commonly used in predictive modeling based on machine learning and deep learning. However, due to the high cost and time required for data acquisition, external model validation is only used in a small percentage of predictive modeling studies. For this reason we propose AdaptiveSplit, a novel approach for predictive modeling studies based on prospective data where the training and external validation sample sizes are adaptively selected throughout the prospective data collection process, rather than being fixed prior to study start. In our analyses, conducted on different datasets, we show how AdaptiveSplit outperforms other splitting methods with pre-defined validation sample size (e.g. pareto split) by maximizing predictive performance and statistical significance even with low sample sizes.
+++

## Introduction
Multivariate predictive models integrate information across multiple variables to construct predictions of a specific outcome and hold promise for delivering more accurate estimates than traditional univariate methods (Woo, 2017, Nat Neurosci). For instance, in case of predicting individual behavioral and psychometric characteristics from brain data, such models can provide higher statistical power and better replicability, as compared to conventional mass-univariate analyses (Spisak et al. 2023). Predictive models can be powered by many different algorithms, from simple linear regression-based models to deep neural networks. With increasing model complexity, the model will be more prone to overfit its training dataset, resulting in biased, overly optimistic in-sample estimates of predictive performance and poor generalizability to data not seen during model fit (ref). Internal validation approaches, like cross-validation (cv) provide means for an unbiased evaluation of predictive performance during model discovery by repeatedly holding out parts of the discovery dataset for testing purposes (refs, e.g. Kohavi R. 1995, Poldrack et al. 2020, JAMA Psychiatry). 

However, internal validation approaches, in practice, still tend to yield overly optimistic performance estimates (Sui et al., 2020, Biol. Psychiatry, Varaquoux, Kaggle paper NPJ dig. med., 2022). There are multiple reasons of this phenomenon. First predictive modelling approaches typically display a high level of "analytical flexibility" and pose a large number of possible methodological choices in terms of feature preprocessing and model architecture, which emerge as uncontrolled "hyperparameters" during model discovery. Seemingly 'innocent' adjustments of such parameters can also lead to overfitting, if it happens outside of the cv loop. The second reason for inflated internally validated performance estimates is 'leakage' of information from the test dataset to the training dataset [refs, e.g. Kapoor \& Narayanan, 2021]. Leakage has many faces, from feature standardization in a non cv-compliant way to co-registration of brain data to a study-specific standard template. Therefore, it is often very hard to notice, especially in complex workflows. Finally, even the highest quality discovery datasets can only yield an imperfect representation of the real world. Therefore, predictive models might capitalize on associations that are specific to the dataset at hand and simply fail to generalize "out-of-the-distribution", e.g. to different populations. Furthermore, models might also be overly sensitive to unimportant characteristics of the training data, like subtle differences between batches of data acquisition or center-effects (Prosperi et al, 2020, Nat Machine Intell.; Spisak, 2022 mlconfound).

The obvious solution for these problems is external validation; that is, to evaluate the model's predictive performance on 'external data' that is guaranteed to be unseen during the the whole model discovery procedure. Acquiring external validation data prospectively, after the final model has already been published [refs], provides the highest possible guarantees for the results' reliability. There is a clear agreement in the community that external validation is critical for establishing machine learning model quality (Collins 2014, BMC Medical Research Methodology; Steyerberg, 2016, J. Clin. Epidemiol.; Ho, 2020, Patterns; Yu, 2022: Radiology: Artificial Intelligence, Spisak et al. 2023 Nature; Poldrack, JAMA Psychiatry, 2020). However, the amount of data to be used for model discovery and external validation is subject of intense discussion [Archer et al. 2020, Riley et al. 2021, both in Statistics in Medicine] [Marek et al. 2022, 2023, Spisak et al. 2023 + some from list here: $https://github.com/spisakt/BWAS_comment$], as such decisions can have crucial implications on the predictive power, replicability and validity of predictive models. Finding the optimal sample sizes is especially challenging for biomedical research, where this trade-off needs to consider both ethical and economic reasons. As a consequence, to date only around 10\% of predictive modeling studies include an external validation of the model (Yang 2022, JAMIA). Those few studies performing true external validation often perform it on retrospective data [Lee et al., Nature Medicicne, 2021] or in a separate, prospective study [Spisak et al., 2020 Nat Comm, Rosenberg attention studies?]. Both approaches can result in a suboptimal use of data and may slow down the dissemination process of new results.

We present a novel adaptive design for predictive modeling studies with prospective data acquisition. With our approach, the sample sizes for training and external validation are not predetermined at the beginning of the study. Instead, they are dynamically determined based on the model's learning curve and the remaining samples' ability to confirm the model's external validity (statistical power). When the model meets predefined conditions during the ongoing data acquisition process, the training procedure concludes, and the model's weights are finalized and publicly deposited, such as through pre-registration. The subsequent data acquisition then serves as a true external validation sample, large enough to powerfully confirm the validity of the model.

## Background

#### The anatomy of an externally validated predictive modelling study

Let us consider the following scenario: a research group plans to involve a fixed number of participants in a study with the aim of constructing a predictive model, and at the same time, evaluate its external validity. How many participants should they allocate for model discovery and how many for external validation to get the highest performing model as well as conclusive validation results?

In most cases it is very hard to make an educated guess about the optimal split of the total sample size into discovery and external validation samples prior to data acquisition. A possible approach is to use simplistic rules of thumb. Splitting data with a 80-20\% ratio (a.k.a Pareto-split) [refs] is probably the most common method, but a 90-10\% or a  50-50\% may also be plausible choices (Rykar \& Saha, 2015, Steyerberg, 2001). However, as illustrated on {numref}`fig1`, such prefixed sample sizes are likely sub-optimal in many cases and the optimal strategy is actually determined by the dependence of the model performance on sample size a.k.a the "learning curve". For instance,  in case of a significant but generally low model performance ({numref}`fig1`A: flat learning curve) the model does not benefit a lot from adding more data to the training set but, on the other hand, it may require a larger external validation set for conclusive evaluation, due to the lower predictive effect size. This is visualized by the "power curve" on {numref}`fig1`, which shows the statistical power of the external validation as a function of sample size used for model discovery. The optimal strategy will be different, however, if the learning curve shows a constant increase, without a strong saturation effect, meaning that predictive performance can be significantly enhanced by training the model on larger samples ({numref}`fig1`B).

:::{figure} figures/fig1.png
:name: fig1
Examples of different optimal discovery and external validation sample sizes compared to a predefined 80-20\% Pareto-split.
(A) If the planned sample size and the model performance is low, the predefined external validation sample size might provide low statistical power to detect a significant model performance. (B) External validation of highly accurate models is well-powered; increasing the training sample size (against the external validation sample size) might result in a better performing final model. (C) Continuing training on the plateau of the learning curve will result in a negligible or biologically not relevant model performance improvement. 
In this case, a larger external validation sample (for more robust external performance estimates) or ‘early stopping’ of the data acquisition process might be desirable
:::

In this case, the stronger predictive performance that can be achieved with larger training sample size, at the same time, allows a smaller external validation sample to be still conclusive.
Finally, in some situations, model performance may rapidly get strong and reach a plateau at a relatively low sample size (Fig. \ref{fig:learning_curves}C). In such cases, the optimal strategy might be to stop early with the discovery phase and allocate resources for a more powerful external validation. 

#### Transparent reporting of external validation: registered models
One of the main criteria of external validation is that the external data has to be independent of the data used during model discovery (Steyerberg 2016, J. Clin. Epidemiol.; etc.). Regardless of the splitting strategy, an externally validated predictive modelling study must provide strong guarantees for this independence criterion. 
In case of prospective studies, this can be realized by means of pre-registering (ref) the study (Fig. \ref{fig:registered_model}A), including the plans for both model discovery and external validation. 

:::{figure} figures/fig2.png
:name: fig2
The ‘adaptive splitting design’ for prospective predictive modeling studies. (A) The procedure with a normal study pre-registration: the model with his parameters is pre-registered and only then it is trained and validated. (B) In our design, during a prospective study the model is trained to fix its parameters and the training sample size. Only then it is registered and subsequently acquired data is used as external validation. (C) The study starts with pre-registering the stopping rules (R1). During the training phase, the candidate models is trained and the splitting rule evaluated, repeatedly as the data acquisition proceeds. When the splitting rule activates, the model is finalized (using the whole training sample) and published/pre-registered (R2). Finally, data acquisition continues and prospective external validation is performed on the newly acquired data.
:::

However, such a direct application of open science practices that have been developed for confirmatory research may not fit well the exploratory nature of the model discovery phase. And alternative approach is to perform the pre-registration phase after model discovery but before the external validation \ref{fig:registered_model}B) (cite: Spisak et al. 2020 Nat Comm, + Balint's PainTone preprint, + other examples, if any). In this case, more freedom is granted for the discovery phase, while the external validation remains equally conclusive, as long as the pre-registration of the external validation includes all details of the finalized model (including the feature pre-processing workflow). This can easily be done by attaching the data and the reproducible analysis code used during the discovery phase or, alternatively, a serialized version of the fitted model (i.e. a file that contains all model weights). We refer such models as "registered models".

#### The adaptive splitting design

Here, we introduce a novel design for prospective predictive modeling studies that leverages the flexibility of model discovery with registered models. Our aim is to determine an optimal splitting strategy, adaptively, during data acquisition. This strategy balances the model performance and the statistical power of the external validation (Fig. \ref{fig:registered_model}C). The proposed design involves continuous model fitting and tuning throughout the discovery phase, for example, after every 10 new participants, and evaluating a 'stopping rule' to determine if the desired compromise between model performance and statistical power of the external validation has been achieved. This marks the end of the discovery phase and the start of the external validation phase, as well as the point at which the model must be publicly and transparently reported/pre-registered. Importantly, the pre-registration should precede the continuation of data acquisition, i.e., the start of the external validation phase.
In the present work, we propose and evaluate a concrete, customizable implementation for the splitting rule. 

## Methods and Implementation

#### Components of the stopping rule


The stopping rule of the proposed adaptive splitting design (for short "AdaptiveSplit") can be formalized as function $S$:

:::{math}
S_\Phi(\mathbf{X}_{act}, \mathbf{y}_{act}, \mathcal{M}) \quad \quad S: \mathbb{R}^2 \longrightarrow \{True, False\}
:::

where $\Phi$ denotes parameters of the rule (to be discussed later), $\mathbf{X}_{act} \in \mathbb{R}^2$ and $\mathbf{y}_{act} \in \mathbb{R}$ is the data (a matrix consisting of $n_{act} > 0$ observations and an fixed number of features $p$) and prediction target, respectively, as acquired so far and $\mathcal{M}$ is the machine learning model to be trained. We aim for constructing a model-agnostic design. The discovery phase ends if and only if the stopping rule returns $True$.

##### **Hard sample size thresholds**

Our stopping rule is designed so that it can force a minimum size for both the discovery and the external validation sample, $t_{min}$ and $v_{min}$, both being free parameters of the stopping rule.

Specifically:

:::{math}
:label: eq-sample-size
    \text{Min-rule:} \quad n_{act} \geq t_{min}
:::

:::{math}
:label: eq-sample-size
    \text{Max-rule:} \quad n_{act} \geq n_{total} – v_{min}
:::

where $n_{act}$ and $n_{total}$ are the actual and total sample sizes, respectively, so that $n_{total} >= n_{act} > 0$.
Setting $t_{min}$ and $v_{min}$ may be useful to prevent early stopping at the beginning of the training procedure, where predictive performance and validation power estimates are not yet reliable due to the small $n{act}$ or to ensure that a minimal validation sample size, even if stopping criteria are never met. If $t_{min}$ and $v_{min}$ are set so that $t_{min} + v_{min} = n_{total}$ then our approach falls back to training a registered model with predefined training and validation sample sizes.

##### **Forecasting Predictive Performance via Learning Curve Analysis**

Taking internally validated  performance estimates of the candidate model as a function of training sample size, also known as learning curve analysis, is  a widely used approach to gain deeper insights into model training dynamics (see examples on Fig. \ref{fig:learning_curves}). In the proposed stopping rule, we will rely on learning curve analysis to provide estimates of the current predictive performance and the expected gain when adding new data to the discovery sample. 

Performance estimates can be unreliable or noisy in many cases, for instance with low sample sizes or when using leave-one-out cross-validation (Varaquoux cv-failure). To obtain stable and reliable learning curves, we propose to calculate multiple (cross-validated) performance estimates from sub-samples sampled without replacement from the actual data set, in order to reduce error. The proposed procedure is detailed in Algorithm \ref{alg:learning-curve}.

:::{prf:algorithm} Bootstrapped Learning Curve Analysis
:label: alg-learning-curve

1. **Require** $\mathbf{X}_{act}, \mathbf{y}_{act}, \mathcal{M}$
2. **Set** $n_b \gets \texttt{<number of bootstrap iterations>}$
3. **For** $t \gets 1$ to $n_{act}$   *(loop over sample sizes)*

    4. **For** $i \gets 1$ to $n_b$ *(bootstrap iterations)*

        5. **Set** $\mathbf{b} \gets$ sample $t$ indices from $<1, \dots, n_{act}>$ without replacement
        6. **Set** $\mathbf{X}_b \gets \mathbf{X}_{act}[\mathbf{b}]$
        7. **Set** $\mathbf{y}_b \gets \mathbf{y}_{act}[\mathbf{b}]$
        8. **Set** $\mathbf{s}[i] \gets$ cross-validated performance score of $\mathcal{M}$ fitted to $(\mathbf{y}_b, \mathbf{X}_b)$

    9. **End For**
    10. **Set** $\textbf{l}_{act}[t] \gets median(\mathbf{s})$

11. **End For**
12. **Return** $\textbf{l}_{act}$  *(bootstrapped learning curve)*
:::

The learning curve analysis allows the discovery phase to be stopped if the expected gain in predictive performance is lower than a predefined relevance threshold and can be used for instance for stopping model training earlier in well-powered experiments and retain more data for the external validation phase. Specifically, the stopping rule $S$ will return $True$ if the \text{Minimu-rule:} is $True$ or the following is true:

:::{math}
:label: eq-perf
    \text{Performance-rule:} \quad \hat{s}_{total} - s_{act} \leq s_{min}
:::

where $s_{act}$ is the actual bootstrapped predictive performance score (i.e. the last element of $\textbf{l}_{act}$, as returned by Algorithm \ref{alg:learning-curve}), $\hat{s}_{total}$ is a estimate of the (unknown) predictive performance $s_{total}$ (i.e. the predictive performance of the model trained the whole sample) and $\epsilon_{s}$ is the smallest predictive effect of interest. Note that, setting $\epsilon_{s} = -\infty$ deactivates the \text{Performance-rule}. 

While $s_{total}$ is typically unknown at the time of evaluating the stopping rule $S$, there are various approaches of obtaining $\hat{s}_{total}$. In the base implementation of AdaptiveSplit, we stick to a simplistic and admittedly overly optimistic method: we extrapolate the learning curve $l_{act}$ based on its tangent line at $n_{act}$, i.e. assuming that the latest growth rate will remain constant for the remaining samples. While in most scenarios this is an unrealistic assumption, it still provides a useful upper bound for the maximally achievable predictive performance with the given sample size and can successfully detect if the learning curve has already reached a flat plateau (like on Fig. \ref{fig:learning_curves}C).

##### **Statistical power of the external validation sample**

Even if the learning curve did not reach a plateau, we still need to make sure that we stop the training phase early enough to save a sufficient amount of data for a successful external validation. Given the actual predictive performance estimate $s_{act}$ and the size of the remaining, to-be-acquired sample $s_{total} - s{act}$, we can estimate the probability that the external validation correctly rejects the null hypothesis (i.e. zero predictive performance). 
This type of analysis, known as power calculation, allows us to determine the optimal stopping point that guarantees the desired statistical power during the external validation.
Specifically, the stopping rule $S$ will return $True$ if the \text{Performance-rule} is $False$ and the following is true:

:::{math}
:label: eq-pow
    \text{Power-rule:} \quad POW_\alpha(s_{act}, n_{val}) \leq v_{pow}
:::

where $POW_\alpha(s, n)$ is the power of a validation sample of size $n$ to detect an effect size of $s$ and $n_{val} = n_{total}-n_{act}$ is the size of the validation sample if stopping, i.e. the number of remaining (not yet measured) participants in the experiment.
Given that machine learning model predictions are often non-normally distributed (Spisak, 2022 mlconfound, ), our implementation is based on a bootstrapped power analysis of permutation tests [refs], as shown in Algorithm \ref{alg:power-rule}. Our implementation is, however, simple to extend with other parametric or non-parametric estimators like Pearson Correlation [refs] and Spearman Rank Correlation [refs].

:::{prf:algorithm} Calculation of the Power-rule
:label: alg-power-rule

1. **Require** $\mathbf{X}_{act}, \mathbf{y}_{act}, n_{validation},  \mathcal{M}, \alpha$
2. **Set** $n_b \gets \texttt{<number of bootstrap iterations>}$
3. **Set** $n_{\pi} \gets \texttt{<number of permutations>}$
4. **Set** $\mathbf{\hat{y}}_{act} \gets \text{cross-validated prediction from } \mathbf{X}_{act} \text{with} \mathcal{M}$

    5. **For** $i \gets 1$ to $n_b$

        6. **Set** $\mathbf{b} \gets$ sample $t$ indices from $<1, \dots, n_{val}>$ with replacement
        7. **Set** $\mathbf{y}_b \gets \mathbf{y}_{act}[\mathbf{b}]$
        8. **Set** $\mathbf{\hat{y}}_b \gets \mathbf{\hat{y}}_{act}[\mathbf{b}]$
        9. **Set** $r_{obs} = correlation( \mathbf{y}_b, \mathbf{\hat{y}}_b)$

        10. **For** $j \gets 1$ to $n_p$

            11. **Set** $\pmb{\pi} \gets$ permute$(<1, \dots, n_{val}>)$
            12. **Set** $\mathbf{y}_{\pi} \gets \mathbf{y}_{b}[\pmb{\pi}]$
            13. **Set** $\mathbf{\hat{y}}_{\pi} \gets \mathbf{\hat{y}}_{b}[\pmb{\pi}]$
            14. **Set** $\mathbf{r}_{null}[j] = correlation( \mathbf{y}_{\pi}, \mathbf{\hat{y}}_{\pi})$

        15. **End For**
        16. **Set** $\mathbf{p}[i] \gets \# (\mathbf{r}_{null} > r_{obs}) / n_{perm}$    

    17. **End For**

18. **Set** $power = \# (\mathbf{p} < \alpha) / n_{b}$
19. **Return** $power$ 
:::

Note that depending on the aim of external validation, the $\text{Power-rule:}$ can be swapped to, or extended with, other conditions. For instance, if we are interested in accurately estimating the predictive effect size, we could condition the stopping rule on the width of confidence interval for the prediction performance. 

Calculating the validation power (Eq. \ref{eq:pow}) for all available sample sizes ($n = 1 \dots n_{act}$) defines the so-called "validation power curve" (see Fig. \ref{fig:learning_curves}), that represents the expected ratio of true positive statistical tests on increasing sample size calculated on the external validation set. Various extrapolations of the power curve can predict the expected stopping point during thee course of the experiment.

#### Stopping Rule

Our proposed stopping rule integrates the $\text{Min-rule}$, the $\text{Min-rule}$, the $\text{Peerformance-rule}$ and the $\text{Power-rule}$ in the following way:

\begin{equation}
    \begin{split}
     S_\Phi(\mathbf{X}_{act}, \mathbf{y}_{act}, \mathcal{M}) = \quad & \\ 
    &\text{Min-rule} \quad AND \\
    & ( \\
    & \quad \text{Max-rule} \quad OR \\
    & \quad\text{Performance-rule} \quad OR \\
    & \quad \text{Power-rule} \\
    &)
    \end{split}
\end{equation}

where $\Phi = <t_{min}, v_{min}, s_{min}, v_{pow}, \alpha>$ are parameters of the stopping rule: minimum training sample size, minimum validation sample size, minimum effect of interest and target power for the external validation and the significance threshold, respectively.

#### AdaptiveSplit package and Rule configuration

In this section, we describe “AdaptiveSplit”, a python package implementing our design and providing a flexible, robust, bootstrapping-based implementation for power calculation and stopping rule evaluation for prospective predictive modeling studies (link to repository). The software can be used together with a wide variety of machine learning tools and provides an easy-to-use interface to work with scikit-learn [refs] models.

#### Empirical evaluation

We evaluate the proposed stopping rule, as implemented in the package \emph{AdaptiveSplit} in four publicly available datasets; the Autism Brain Imaging Data Exchange (ABIDE) [refs], the Human Connectome Project [refs], the Information eXtraction from Images (IXI) [refs] and the Breast Cancer Wisconsin (BCW) [refs] datasets (Fig. 3). 

##### **ABIDE**
> add a few-sentence description to all datasets, including type of data, number of features target variable, sample size, etc...

##### **HCP**
> todo

##### **IXI**
> todo

##### **BCW**
> todo

The chosen datasets include both classification and regression tasks, and span a wide range in terms the number of participants, number of predictive features, achievable predictive effect size (new Fig: learning and power curves for each dataset) and data homogeneity.
Our analyses aim to contrast the proposed adaptive splitting method to the application of fixed training and validation sample sizes, specifically to using 50, 60 (a.k.a Pareto-split) or 90\% of the total sample size for training and the rest for external validation.
We simulate various total sample sizes ($n_{total}$) with random sampling without replacement. 
For a given total sample size, we have simulated the prospective data acquisition procedure by incrementing $n_{act}$; starting with 10\% of the total sample size and going up with increments of five. In each step, the stopping rule was evaluated with "AdaptiveSplit", but fitting a Ride model (for regression tasks) or a L2-regularized logistic regression (for classification tasks). Model fit always consisted of a cross-validated fine-tuning of the regularization parameter (list values), resulting in bootstrapping a nested cross-validation-based estimate of prediction performance and validation power (see Algorithms \ref{alg:learning-curve} and \ref{alg:power-rule}).
This procedure was iterated until the stopping rule returned True. The corresponding sample size was then considered thee final training sample.
With all four splits approaches (adaptive, Pareto, Half-split, 90-10\% split), we trained the previously described Ridge or regularized logistic regression model  on the training sample and obtained predictions for the sample left out for external validation. 
This whole procedure is repeated 100 times for each (simulated) total sample size in each dataset (approximately X splitting rule evaluations), to estimate the confidence intervals for the models performance in the external validation and its statistical significance.
In all analyzes, the adaptive splitting procedure is performed with a target power of $v_{pow} = 0.8$, an $alpha = 0.05$,  $t_{tmin} = n_{total}/3$, $v_{min}=12$, $s_{min}=-\infty$.  P-values were calculated using a permutation test with 5000 permutations. 

## Results

The proposed adaptive splitting technique for prospective predictive modelling studies has been implemented in the the Python package "AdaptiveSplit" (link).

The results of our empirical analysis confirmed on four large, openly available datasets that the proposed adaptive splitting approach can successfully identify the optimal time to stop acquiring data for training and maintain a good compromise between maximizing both predictive performance and external validation power with any "sample size budget".

In all three samples, the applied models yielded a statistically significant predictive performance at much lower sample sizes than the total size of the dataset, i.e. all datasets were well powered for the analysis. The models explained X\% of the variance in cognitive abilities based on functional brain connectivity in the HCP dataset and Y\% in age, based on structural MRI data (gray matter probability maps) in the IXI dataset. Classification accuracy was X for autism diagnosis (functional connectivity) in the ABIDE dataset and Y for breast cancer diagnosis in the BCW dataset (type of data).

The datasets varied not only in the achievable predictive performance but also in the shape of the learning curve, with different sample sizes (Fig) and thus, they provided a good opportunity to evaluate the performance of our stopping rule in various circumstances.

We found that adaptively splitting the data provided external validation performances that were comparable to the commonly used Pareto split (80-20\%) in most cases (Fig. \ref{fig:results} left column). As expected half-split tended to provide worse predictive performance due to the smaller training sample . In contrast, 90-10\% tended to display only slightly higher performances than the Pareto and the Adaptive splitting techniques.
This small achievement came with a big cost in terms of the statistical power in the external validation sample, where the 90-10\% split very often gave inconclusive results (p$\geq$0.05) (Fig. \ref{fig:results} right column).
Although to a lesser degree, Pareto split also frequently failed to yield a conclusive external validation. Adaptive splitting (as well as half-split) provided sufficient statistical power for the external validation in most cases.

In sum, when contrasting splitting approaches based on fixed validation size with adaptive splitting, using the latter was always the preferable strategy to maximize power and statistical significance during external validation. The benefit of adaptively splitting the data acquisition for training and validation provides the largest benefit in lower sample size regimes.
In case of larger sample sizes, the fixed Pareto split (20-80\%) provided also good results, giving similar external validation performances to adaptive splitting.

:::{figure} figures/fig3.png
:name: fig3
Comparison of splitting methods on external validation performance (left column) and p-values (right column) at various $n_{total}$. We see how the AdaptiveSplit's overall performance is comparable to Pareto's on each of the three neuroimaging datasets, albeit adaptively splitting data works better than pareto at low sample sizes, where external validation performance can be slightly lower but statistical significance is always higher. We can also observe how AdaptiveSplit outperforms 50-50 and 90-10 splits, which have low external validation scores and statistical significance, respectively.
:::

## Discussion

Adaptively splitting data allows to keep the external validation performance "in check" by determining the best trade-off between model performance and statistical validity, allowing the generation of reliable results even with  smaller sample sizes. This feature makes our design a strong choice for developing dependable and repeatable models for prospective data acquisition and model registration purposes. Model finalization is facilitated by adaptively setting training and external validation sample sizes, which saves time for acquisition and publication while simultaneously increasing model dependability when the it is appropriately registered. Including aspects that assure a good final model, such as data splitting, cross-validation, and bootstrapping for robust performance estimation, the adaptive split design makes it simple to avoid general pitfalls in predictive modeling, e.g. overfitting, data leakage or lack of reproducibility. Indeed, as a result of its bootstrap-based learning and power curves estimates on which the stopping rule is evaluated, the adaptive split method enables results' reproducibility on the same data and seems to give comparable results on different datasets when the same set of parameters is used (see Fig. 3).

- Issues with BWAS replication, cv-failure
- How to design studies for predictive model discovery?
    registered models
- When can an adaptive splitting design be beneficial
- What's with Pareto and others?
- What if we can adjust ext. val sample?

Outlook:
- how adaptiveesplit pushes forward the filed
- varieties of the stopping rule, for various questions...





## Conclusion
Model registration for predictive modeling studies is an essential opportunity to improve the credibility and reproducibility of scientific works based on predictive modeling, since evidences by recent studies show how predictive models are often misused and/or misspecified (Kapoor \& Narayanan, 2021, other refs?). We presented the adaptive split design, a methodology allowing the examination of models' performance during prospective data acquisition in order to fix and finalize model for registration purposes. When we compare the adaptive training/external validation splitting strategy to more commonly used splitting methods, we find that flexibility in the definition of training / external validation sample size allows to deliver a model that is able to find the best trade-off between predictive performance and statistical significance even when a large sample size isn't available, making the definition of an early model possible while data collection is still ongoing and so enhancing transparency and study repeatability. Model registration for predictive modeling-based study design may boost trust in model creation and performance in several disciplines of research, particularly those that include the prediction of complex outcomes as, for instance, biomedical science.

