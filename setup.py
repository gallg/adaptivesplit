from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'AdaptiveSplit'
LONG_DESCRIPTION = 'External validation of machine learning models with adaptive sample splitting'

# Setting up
setup(
    name="adaptivesplit",
    version=VERSION,
    author="PNI Lab (Predictive NeuroImaging Laboratory)",
    author_email="<giuseppe.gallitto@uk-essen.de>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["joblib==1.2.0", "matplotlib==3.6.0", "numpy==1.24.2", "pandas==1.5.1", "pygam==0.9.0",
                      "regressors==0.0.3", "scikit_learn==1.2.1", "scipy==1.11.3", "tqdm==4.64.1", "setuptools==57.5.0"],
    keywords=['python', 'machine learning', 'neuroimaging', 'data splitting'],
    classifiers=[]
)
