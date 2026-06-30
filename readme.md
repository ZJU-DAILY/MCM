# Core Maintenance on Dynamic Multilayer Graphs

## Introduction

This repository contains the source code for Core Maintenance on Dynamic Multilayer Graphs. The baseline of decomposition used for computing MCVs (under the ``MlcDec/`` directory) can be found [here](https://github.com/MDCGraph/MlcDec/) [1]. 


*[1] Dandan Liu, Run-An Wang, Zhaonian Zou, and Xin Huang. 2024. Fast Multilayer  Core Decomposition and Indexing. In ICDE. 2695–2708.*

## Dataset

The datasets used in our experiments can be found [here](https://drive.google.com/drive/folders/1owcVszSN9wbyDSmh2h0a4nJa_m8tUmKk?usp=sharing
). Please download the datasets and place them under the ``MlcDec/datasets/layer`` directory.

## Usage

Compile the project:

```bash
cd src
```

Compile the project:

```bash
make
```

Run baseline and maintenance algorithm:


```bash
./run_baseline <dataset_path> <num_edges> [num_threads]
./run_test <dataset_path> [num_threads]
```


Example: run batch-edge maintenance algorithm with 500 updated edges, and compare against the baseline



```bash
./run_baseline ../MlcDec/datasets/layer/obamainisrael/ 500
./run_test ../MlcDec/datasets/layer/obamainisrael/
```

Example: run single-edge maintenance algorithm, and compare against the baseline




```bash
./run_baseline ../MlcDec/datasets/layer/obamainisrael/ 1
./run_test ../MlcDec/datasets/layer/obamainisrael/
```

Clean generated binaries and object files:

```bash
make clean
```

## Requirements

GCC with C++17

OpenMP

Linux
