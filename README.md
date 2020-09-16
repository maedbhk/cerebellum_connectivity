cerebellum_connectivity 
==============================

Install `pyenv` using Homebrew:

    $ brew update
    $ brew install pyenv

Add `pyenv init` to your shell:

    $ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
    $ source ~/.bash_profile

Install the required version of python:

    $ pyenv install 3.7.0

### Installing the Required Python Packages

This project uses [`pipenv`](https://github.com/pypa/pipenv) for virtual environment and python package management.

Ensure pipenv is installed globally:

    $ brew install pipenv

Navigate to the top-level directory in 'cerebellum_connectivity' and install the packages from the `Pipfile.lock`.
This will automatically create a new virtual environment for you and install all requirements using the correct version of python.

    $ pipenv install

## Activating the virtual environment:

    $ pipenv shell

> NOTE: To deactivate the virtual environment when you are done working, simply type `exit`

## Running connectivity models:

    $ from connectivity.models.run_connectivity import train_evaluate
    $ train_evaluate()

> NOTE: this command does model training and evaluation with the default parameters set in `model_config.json`
> to provide different arguments to the training and/or evaluation routines, pass in parameters from `model_config.json` > as key-word args

    $ from connectivity.models.run_connectivity import train_evaluate
    $ train_evaluate(lambdas=[10, 100, 1000], train_subjects=[3, 4], eval_subjects=[3, 4])


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` (not implemented)
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── sc1            <- Data from study 1
    │       └── anatomicals          
    │       └── GLM_firstlevel_7
    │       └── GLM_firstlevel_8
    │       └── imaging_data
    │       └── conn_models
    │       └── encoding
    │       └── suit
    │       └── surfaceFreesurfer
    │       └── surfaceWB
    │       └── beta_roi
    │   └── sc2            <- Data from study 2         
    │       └── GLM_firstlevel_7
    │       └── GLM_firstlevel_8
    │       └── conn_models
    │       └── encoding
    │       └── suit
    │       └── beta_roi    
    │   └── tasks.json           <- contains information about tasks across studies
    │
    ├── docs                     <- A default Sphinx project; see sphinx-doc.org for details (not yet implemented)
    │
    │
    ├── notebooks                <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                               the creator's initials, and a short `-` delimited description, e.g.
    │                               `1.0-mk-visualize_maps`.
    │
    ├── references               <- Manuals, and all other explanatory materials (not yet implemented)
    │
    ├── reports                  <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures              <- Generated graphics and figures to be used in reporting
    │
    ├── Pipfile                  <- The Pipfile for reproducing the analysis environment, e.g.
    │                               install all packages with `pipenv install`, check existing packages `pipenv graph`
    │
    ├── setup.py                 <- makes project pip installable (pip install -e .) so connectivity package can be imported
    ├── connectivity             <- Source code for use in this project.
    │   ├── __init__.py          <- Makes connectivity a Python module
    │   ├── constants.py         <- Default directories   
    │   ├── io.py                <- Import/Output .mat, .h5, .json files
    │   │
    │   ├── data                 <- Scripts to generate data for modelling, currently implemented for betas only
    │   │   └── prep_data.py
    │   │
    │   ├── features             <- Scripts to turn raw data into features for modeling (not yet implemented)
    │   │   └── build_features.py
    │   │
    │   ├── models               <- Scripts to train models and then use trained models to make
    │   │   │                       predictions. Parameters are set in `model_config.json`
    │   │   ├── run_connectivity.py
    │   │   └── train_model.py
    │   │   └── evaluate_model.py
    │   │   └── model_functions.py
    │   │   └── model_config.json
    │   │   
    │   ├── scripts              <- Example bash script for training and evaluating models
    │   │   ├── run_connectivity.sh
    │   │
    │   └── visualization        <- Scripts to create exploratory and results-oriented visualizations. Parameters are set in `visualize_config.json`
    │       └── visualize.py
    |       └── image_utils.py
            └── visualize_config.py
    │
    └── tox.ini                 <- tox file with settings for running tox; see tox.testrun.org (not implemented)


--------

