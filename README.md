# QuGStep
![GitHub](https://img.shields.io/github/license/gbup-group/DIANet.svg)

By [Senwei Liang](https://leungsamwai.github.io) <sup>1†</sup>, [Linghua Zhu](https://scholar.google.com/citations?user=BNPyHf4AAAAJ&hl=en) <sup>1†</sup>, Xiaosong Li and Chao Yang

This repo is the implementation of "QuGStep: Refining Step Size Selection in Gradient Estimation for Variational Quantum Algorithms" [[paper]](https://arxiv.org/abs/2503.14366).

## Introduction

Finite expression method (FEX) is a new methodology that seeks an approximate PDE solution in the space of functions with finitely many analytic expressions. This repo provides a deep reinforcement learning method to implement FEX for various high-dimensional PDEs in different dimensions.

![image](fexrl.png)

## Environment
* Qskit

## Code structure

```
Finite-expression-method
│   README.md    <-- You are here
│
└─── fex    ----> three numerical examples with FEX
│   │   Poisson
│   │   Schrodinger
│   │   Conservationlaw
│   
└─── nn     ----> three numerical examples with NN
    │   Poisson
    │   Schrodinger
    │   Conservationlaw
```
## Citing QuGStep
If you find the code in this repo is helpful for your research, please kindly cite
```
@article{liang2025qugstep,
  title={QuGStep: Refining Step Size Selection in Gradient Estimation for Variational Quantum Algorithms},
  author={Liang, Senwei and Zhu, Linghua and Li, Xiaosong and Yang, Chao},
  journal={arXiv preprint arXiv:2503.14366},
  year={2025}
}
```
