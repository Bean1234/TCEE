# Inductive Reasoning with Type-Constrained Encoding for Emerging Entities

This is the official code release of the following paper:

Inductive Reasoning with Type-Constrained Encoding for Emerging Entities.  Accepted by Neural Network.

## Quick Start

### Installation

Run the following commands to install the required packages:

```
pip install -r requirements.txt
```

### Dataset

```
unzip data.zip
```

It will generate three dataset folders in the ./data directory. In our experiments, the datasets used are: `WN-MBE`, `FB-MBE`, and `NELL-MBE`.
In each dataset, there are six `add_x` folders, where `add_1` is the validation set of the original KG, and `add_2~6` are emerging batches.

### Training and evaluation

1. Train model

   During training, you can use the `--run_analysis` to see the test results on emerging batches.
   The model will be evaluated on both validation and emerging batch testing sets during training.
   Note that due to the different densities of entity links, there will be a gap between the results on the validation and test sets.

```
python src/experiments.py --train --run_analysis --dataset <dataset-name> --gpu <gpu-ID>
```

2. Evaluate model

```
python src/experiments.py --inference --dataset <dataset-name> --gpu <gpu-ID>
```

You can use the following datasets: `WN-MBE`, `FB-MBE`, and `NELL-MBE`.

### Change the hyperparameters

To change the hyperparameters and other experiment setups, start from the [parse_args files](src/parse_args.py) and [config files](src/config.py).

## Acknowledgement

We refer to the code of [MBE](https://github.com/nju-websoft/MBE). Thanks for their contributions.
