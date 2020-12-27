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

Navigate to the top-level directory in `cerebellum_connectivity` and install the packages from the `Pipfile.lock`.
This will automatically create a new virtual environment for you and install all requirements using the correct version of python.

    $ pipenv install

## Activating the virtual environment:

    $ pipenv shell

> NOTE: To deactivate the virtual environment when you are done working, simply type `exit`

## Running connectivity models:

    And example of how to run a set of connectivity models + evaluate them can be found in 
    Scripts/sript_ridge.py  


Project Organization
------------

Data Organization: 
------------
    ├── data
    │   └── sc1            <- Data from study 1
    │       └── anatomicals          
    │       └── GLM_firstlevel_7
    │       └── GLM_firstlevel_8
    │       └── imaging_data
    │       └── conn_models
    │       └── suit
    │       └── surfaceFreesurfer
    │       └── surfaceWB
    │       └── beta_roi
    │   └── sc2            <- Data from study 2         
    │       └── GLM_firstlevel_7
    │       └── GLM_firstlevel_8
    │       └── conn_models
    │       └── suit
    │       └── beta_roi    

Repository structure:
------------ 
    ├── LICENSE
    ├── Makefile                 <- Makefile with commands like `make data` (not implemented)
    ├── README.md                <- The top-level README for developers using this project.
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
    │   ├── constants.py         <- Global defauls (not a class!) and default directories   
    │   ├── io.py                <- Module for Import/Output .mat, .h5, .json files
    │   ├── data.py              <- Module containing the Dataset class 
    │   ├── models.py            <- Module for defining different models classes (mostly derived from sklearn, but potentially with mixin )
    |   ├── run.py               <- Module defining function to traind and evaluate a model on a set of subjects - uses `train_config` and `eval_config` to set the parameters
    │   └── visualize.py         <- Module to map results to the surface and visualize them. Parameter should be set as globals in module - not a class! 
--------

