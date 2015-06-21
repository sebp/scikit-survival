# Fast Training of Support Vector Machines for Survival Analysis

This repository contains an efficient implementation of *Survival Support
Vector Machines* as proposed in

> Pölsterl, S., Navab, N., and Katouzian, A.,
> *Fast Training of Support Vector Machines for Survival Analysis*,
> In Proceedings of the European Conference on Machine Learning and
> Principles and Practice of Knowledge Discovery in Databases (ECML PKDD), 2015

## Requirements

- Python 3.2 or later
- numexpr
- numpy 1.9 or later
- pandas 0.15.2 or later
- scikit-learn 0.16
- scipy 0.15 or later
- six
- C/C++ compiler

## Getting Started

The easiest way to get started is to install (https://store.continuum.io/cshop/anaconda/)[Anaconda]
and setup an environment. To create a new environment, named `ssvm`, run:

```
conda create -n ssvm python=3 --file requirements.txt
```

To work in this environment, ``activate`` it as follows:

```
source activate ssvm
```

If you are on Windows, run the above command without the ``source`` in the beginning.

Once you setup your build environment, you have to compile the C/C++
extensions and install the package by running:

```
python setup.py install
```

Alternatively, if you want to use the package without installing it,
you can compile the extensions in place by running:

```
python setup.py build_ext --inplace
```

To check everything is setup correctly run the test suite by executing:

```
nosetests
```


## Datesets

The repository contains 4 out of 6 datasets that are freely available and can
be used to reproduce the results in the paper.

| Dataset | Samples | Features | Events | Outcome |
| ------- | ------- | -------- | ------ | ------- |
| [AIDS study][Hosmer2008] | 1,151 | 13 | 96 (8.3%) | AIDS defining event or death |
| [Breast cancer][Desmedt2007] | 198 | 80 | 62 (31.3%) | Distant metastases |
| [Veteran's Lung Cancer][Kalbfleisch2008] | 137 | 6 | 128 (93.4%) | Death |
| [Worcester Heart Attack Study][Hosmer2008] | 500 | 14 | 215 (43.0%) | Death|



[Desmedt2007]: http://dx.doi.org/10.1158/1078-0432.CCR-06-2765 "Desmedt, C., Piette, F., Loi et al.: Strong Time Dependence of the 76-Gene Prognostic Signature for Node-Negative Breast Cancer Patients in the TRANSBIG Multicenter Independent Validation Series. Clin. Cancer Res. 13(11), 3207–14 (2007)"

[Hosmer2008]: http://www.wiley.com/WileyCDA/WileyTitle/productCd-0471754994.html "Hosmer, D., Lemeshow, S., May, S.: Applied Survival Analysis: Regression Modeling of Time to Event Data. John Wiley & Sons, Inc. (2008)"

[Kalbfleisch2008]: http://www.wiley.com/WileyCDA/WileyTitle/productCd-047136357X.html "Kalbfleisch, J.D., Prentice, R.L.: The Statistical Analysis of Failure Time Data. John Wiley & Sons, Inc. (2002)"
