# KDI-feature-selection

The codes serves as an example to demonstrate the effectiveness of the KDI feature selection method in paper "A Kernel Discriminant Information Approach to Nonlinear Feature Selection".

# Environments required

- Numpy
- Tensorflow
- scikit-learn
- Scipy


# Running the examples

We demonstrate the effectiveness of the proposed approach on open-access benchmark feature selection datasets, i,e, SMK_CAN_187 and TOX_171.

```shell
git clone https://github.com/zejiangh/KDI-feature-selection

python example.py --option SMK_CAN_187.mat

python example.py --option TOX_171.mat

```
