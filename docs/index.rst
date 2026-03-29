
.. _index:


=======
dtangle
=======

Python implementation of dtangle for deconvolution of bulk RNA-seq and microarray mixtures.


Get started
-----------

You can install ``dtangle`` with ``pip``::

    pip install dtangle

Minimal example::

    from dtangle import deconvolut
    deconvolut(adata, adata_references, "cell_type")


For more details, see :ref:`install`.


Contact
-------

If you found a bug, please use the `Issue tracker <https://github.com/harryhaller001/dtangle/issues>`_.




.. toctree::
    :caption: Start
    :maxdepth: 4
    :glob:

    install


.. toctree::
    :caption: Tutorials
    :maxdepth: 4
    :glob:

    notebooks/vignette_basic
    notebooks/vignette_brain
    notebooks/vignette_pbmc3k


.. toctree::
    :caption: API Documentation
    :maxdepth: 4
    :glob:

    autoapi/dtangle/index
