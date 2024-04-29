---
exports:
  - format: docx
    output: exports/adaptivesplit_manuscript_si.docx
---
# Supplementary Material

## Supplementary Figures


:::{figure} figures/si-bcw-confmat.png
:name: si-bcw-confmat
:align: center
:width: 40%
Predictive performance (confusion matrix) of the model trained on the BCW dataset to predict diagnosis. The model was trained on the whole dataset with nested cross-validation.
:::

:::{figure} figures/si-bcw-lc.png
:name: si-bcw-lc
:align: center
:width: 50%
Learning curve (top) and power curve (bottom) of the model trained on the BCW dataset to predict diagnosis. The maximum sample size (i.e. the whole dataset) was considered as the "sample size budget". X-axis: $n_{act}$; y-axis (learning curve): Accuracy as a measure of predictive performance; y-axis (power curve): statistical power of the remaining sample to confirm the model's validity.
:::

:::{figure} figures/si-ixi-scatter.png
:name: si-ixi-scatter
:align: center
:width: 50%
Predictive performance of the model trained on gray matter probability images from the IXI dataset to predict age. The model was trained on the whole dataset with nested cross-validation. X-axis: true age, y-axis: predicted age.
:::

:::{figure} figures/si-ixi-lc.png
:name: si-ixi-lc
:align: center
:width: 50%
Learning curve (red) and power curve (blue) of the model trained on gray matter probability images from the IXI dataset to predict age. The maximum sample size (i.e. the whole dataset) was considered as the "sample size budget". X-axis: $n_{act}$; y-axis (learning curve): Pearson's correlation as a measure of predictive performance; y-axis (power curve): statistical power of the remaining sample to confirm the model's validity.
:::

:::{figure} figures/si-hcp-scatter.png
:name: si-hcp-scatter
:align: center
:width: 50%
Predictive performance of the model trained on resting state functional connectivity data from the HCP dataset to predict fluid intelligence (PMAT24_A_CR). The model was trained on the whole dataset with nested cross-validation. X-axis: true age, y-axis: predicted age.
:::


:::{figure} figures/si-hcp-lc.png
:name: si-hcp-lc
:align: center
:width: 50%
Learning curve (red) and power curve (blue) of the model trained on resting state functional connectivity data from the HCP dataset to predict fluid intelligence (PMAT24_A_CR). The maximum sample size (i.e. the whole dataset) was considered as the "sample size budget". X-axis: $n_{act}$; y-axis (learning curve): Pearson's correlation as a measure of predictive performance; y-axis (power curve): statistical power of the remaining sample to confirm the model's validity.
:::

## Supplementary Tables

### Supplementary Table 1
Manuscripts, commentaries, and editorials on the topic of brain-behavior associations and their reproducibility, related to [](https://doi.org/10.1038/s41586-022-04492-9). See the up-to-date list here: https://spisakt.github.io/BWAS_comment/

| Authors                           | Title                                                                                                      | Where                                                                                                               |
|-----------------------------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Nature editorial                | Cognitive neuroscience at the crossroads                                                                  | [Nature](https://www.nature.com/articles/d41586-022-02283-w)      
|  Spisak et al.                  |      Multivariate BWAS can be replicable with moderate sample sizes | [Nature](https://doi.org/10.1038/s41586-023-05745-x) |
| Nat. Neurosci. editorial        | Revisiting doubt in neuroimaging research                                                                  | [Nat. Neurosci.](https://doi.org/10.1038/s41593-022-01125-2)                                                |
| Monica D. Rosenberg and Emily S. Finn | How to establish robust brain–behavior relationships without thousands of individuals                      | [Nat. Neurosci.](https://doi.org/10.1038/s41593-022-01110-9)                                                |
| Bandettini P et al.               | The challenge of BWAS: Unknown Unknowns in Feature Space and Variance                                      | [Med](http://www.thebrainblog.org/2022/07/04/the-challenge-of-bwas-unknown-unknowns-in-feature-space-and-variance/) |
| Gratton C. et al.                 | Brain-behavior correlations: Two paths toward reliability                                                  | [Neuron](https://doi.org/10.1016/j.neuron.2022.04.018)                                                |
| Cecchetti L. and Handjaras G.     | Reproducible brain-wide association studies do not necessarily require thousands of individuals     | [psyArXiv](10.31234/osf.io/c8xwe)                                                                              |
| Winkler A. et al.                 | We need better phenotypes                                                                                  | [brainder.org](https://brainder.org/2022/05/04/we-need-better-phenotypes/)                                          |
| DeYoung C. et al.                 | Reproducible between-person brain-behavior associations do not always require thousands of individuals                              | [psyArXiv](10.31234/osf.io/sfnmk)                                                                              |
| Gell M et al.                     | The Burden of Reliability: How Measurement Noise Limits Brain-Behaviour Predictions                        | [bioRxiv](https://doi.org/10.1101/2023.02.09.527898)                                             |
| Tiego J. et al.                   | Precision behavioral phenotyping as a strategy for uncovering the biological correlates of psychopathology | [OSF](10.31219/osf.io/geh6q)                                                                                        |
| Chakravarty MM.                   | Precision behavioral phenotyping as a strategy for uncovering the biological correlates of psychopathology | [Nature Mental Health](https://doi.org/10.1038/s44220-023-00057-5)            | 
| White T.                       | Behavioral phenotypes, stochastic processes, entropy, evolution, and individual variability: Toward a unified field theory for neurodevelopment and psychopathology | [OHBM Aperture Neuro](https://doi.org/10.52294/c900ce20-3ffd-4545-8c15-3ec532b2ee3b)            | 
| Bandettini P.                   | Lost in transformation: fMRI power is diminished by unknown variability in methods and people | [OHBM Aperture Neuro](10.52294/725139d7-0b8a-49dc-a81d-ba2ca64ff6d9)            | 
| Thirion B.                      | On the statistics of brain/behavior associations | [OHBM Aperture Neuro](10.52294/51f2e656-d4da-457e-851e-139131a68f14)           | 
| Tiego J., Fornito A.                 | Putting behaviour back into brain–behaviour correlation analyses | [OHBM Aperture Neuro](10.52294/2f9c5854-d10b-44ab-93fa-d485ef5b24f1)            | 
| Lucina QU.                     | Brain-behavior associations depend heavily on user-defined criteria | [OHBM Aperture Neuro](https://doi.org/10.52294/5ba14033-72bb-4915-81a3-fa221302818a) | 
| Valk SL., Hettner MD.                 |   Commentary on ‘Reproducible brain-wide association studies require thousands of individuals’ | [OHBM Aperture Neuro](10.52294/de841a29-d684-4707-9042-5bbd3d764f84)            | 
| Kong XZ., et al.                | Scanning reproducible brain-wide associations: sample size is all you need? | [Psychoradiology](https://doi.org/10.1093/psyrad/kkac010)            | 
| J. Goltermann, et al. | Cross-validation for the estimation of effect size generalizability in mass-univariate brain-wide association studies | [BioRxiv](https://doi.org/10.1101/2023.03.29.534696) | 
| Kang K., et al. | Study design features that improve effect sizes in cross-sectional and longitudinal brain-wide association studies | [BioRxiv](https://doi.org/10.1101/2023.05.29.542742) |
| Makowski C., et al. | Reports of the death of brain-behavior associations have been greatly exaggerated |[BioRxiv]( https://doi.org/10.1101/2023.06.16.545340) |
| J. Wu et al.  | The challenges and prospects of brain-based prediction of behaviour | [Nat. Human Behaviour](https://doi.org/10.1038/s41562-023-01670-1) | 

:::