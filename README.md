# `gp_grief`
Large Scale Gaussian Processes using Grid-Structured Eigenfunction (GRIEF) kernels.

# Installation & Setup
Only python 2 is supported.
An up-to-date Anaconda distribution is recommended in addition to the following non-standard dependencies (using `pip`):
```
pip install tabulate
pip install GPy # v1.6.1 tested but other versions should work
```
Also required is the forked version of the `py-mcmc` library:
```
git clone https://github.com/treforevans/py-mcmc.git
cd py-mcmc
python setup.py install
```

# Tutorials & Examples
* [Type-II inference with GRIEF kernels:](./tutorials/Type-II example with GRIEF kernel.ipynb) This example demonstrates the type-II training and inference procedure with GP-GRIEF.
* [Type-I inference with GRIEF kernels:](./tutorials/Type-I example with GRIEF kernel) This example demonstrates how to perform fully Bayesian type-I inferece with MCMC in a time complexity independent of the quantity of training data.

# Citation
The underlying algorithms are based on the 2018 ICML paper:

```
@InProceedings{evans_gp-grief,
  title = 	 {Scalable {G}aussian Processes with Grid-Structured Eigenfunctions ({GP}-{GRIEF})},
  author = 	 {Evans, Trefor and Nair, Prasanth},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {1416--1425},
  year = 	 {2018}
}
```
