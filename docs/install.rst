
.. _install:


============
Installation
============

.. highlight:: console
.. _setuptools: https://pypi.org/project/setuptools/


For installation of this package you need to have Python 3.11 or newer installed. You can install ``dtangle`` with ``pip``::

    pip install dtangle


Development
-----------

Install development version of `dtangle` with::

    git clone https://github.com/harryhaller001/dtangle.git
    cd dtangle


To setup development environment create python virtual environment::

    uv venv


Use `make` to setup dependencies::

    # Install dependencies and activate pre-commit hook
    make install


Run checks with `make`::

    # Run all checks
    make check

    # Or run individial check functions

    # Run formatting
    make format

    # Run unit testing with pytest
    make testing

    # Run type checks with mypy
    make typing
