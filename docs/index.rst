.. USB documentation master file, created by
   sphinx-quickstart on Mon Aug 22 17:27:31 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to USB's documentation!
===============================
Unified Semi-supervised learning Benchmark (USB) is a modular and extensible codebase including data pipeline, and popular SSL algorithms for standardization of SSL ablations. 
Meanwhile, pre-trained versions of the state-of-the-art neural models for CV tasks are provided.
It is easy-to-use/extend, affordable, and comprehensive for developing and evaluating SSL algorithms. 
USB provides the implementation of 14 SSL algorithms based on Consistency Regularization, and 15 tasks for evaluation from CV, NLP, and Audio domain.


We provide a Python package *semilearn* of USB for users who want to start training/testing the supported SSL algorithms on their data quickly:

```
pip install semilearn
```

For detailed information on the various tasks over different domains, the developed algorithms and the customized usage, please refer to the following sections. 

.. toctree::
   :maxdepth: 2

   
   Datasets Zoo <benchmark>
   Algorithm Zoo <algorithm_zoo>
   Tutorial <tutorial>
   API <api>


   :caption: Contents:



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
