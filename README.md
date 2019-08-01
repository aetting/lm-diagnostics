# LM diagnostics

This repository contains the diagnostic datasets and experimental code for *What BERT is not: Lessons from a new suite of psycholinguistic diagnostics for language models*, by Allyson Ettinger.

# Diagnostic test data

The `datasets` folder contains TSV files with data for each diagnostic test, along with explanatory README files for each dataset.

(Dataset-specific README files are still in progress.)

# Code

The code in this section can be used to process the datasets for input to a language model, and to run the diagnostic tests on that language model's predictions. It should be used in three steps:

### Step 1) Process datasets to produce inputs for LM

`proc_datasets.py` can be used to process the provided datasets into 1) `<testname>-contextlist` files containing contexts (one per line) on which the LM's predictions should be conditioned, and b) `<testname>-targetlist` files containing target words (one per line, aligned with the contexts in `*-contextlist`) for which you will need probabilities conditioned on the corresponding contexts. Repeats in `*-contextlist` are intentional, to align with the targets in `*-targetlist`.

Basic usage:
```
python proc_datasets.py \
  --outputdir <location for output files> \
  --role_stim datasets/ROLE-88/ROLE-88.tsv \
  --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
  --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
  --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv \
  --add_mask_tok
```
* `add_mask_tok` flag will append '[MASK]' to the contexts in `*-contextlist`, for use with BERT.
* `<testname>` comes from the following list: *cprag*, *role*, *negsimp*, *negnat* for CPRAG-34, ROLE-88, NEG-88-SIMP and NEG-88-NAT, respectively.

### Step 2) Get LM predictions/probabilities

You will need to produce two files: one containing top word predictions conditioned on each context, and one containing the probabilities for each target word conditioned on its corresponding context.

**Predictions**: Model word predictions should be written to a file with naming `modelpreds-<testname>-<modelname>`.  Each line of this file should contain the top word predictions conditioned on the context in the corresponding line in `*-contextlist`. Word predictions on a given line should be separated by whitespace. Number of predictions per line should be no less than the highest *k* that you want to use for accuracy tests.

**Probabilities** Model target probabilities should be written to a file with naming `modeltgtprobs-<testname>-<modelname>`. Each line of this file should contain the probability of the target word on the corresponding line of `*-targetlist`, conditioned on the context on the corresponding line of `*-contextlist`.

* `<testname>` list is as above. `<modelname>` should be the name of the model that will be input to the code in Step 3.

### Step 3) Run accuracy and sensitivity tests for each diagnostic

`prediction_accuracy_tests.py` takes `modelpreds-<testname>-<modelname>` as input and runs word prediction accuracy tests.

Basic usage:

```
python prediction_accuracy_tests.py \
  --preddir <location of modelpreds-<testname>-<modelname>> \
  --resultsdir <location for results files> \
  --models <names of models to be tested, e.g., bert-base-uncased bert-large-uncased> \
  --k_values <list of k values to be tested, e.g., 1 5> \
  --role_stim datasets/ROLE-88/ROLE-88.tsv \
  --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
  --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
  --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv
```

`sensitivity_tests.py` takes `modeltgtprobs-<testname>-<modelname>` as input and runs sensitivity tests.

Basic usage:
```
python sensitivity_tests.py \
  --probdir <location of modelpreds-<testname>-<modelname>> \
  --resultsdir <location for results files> \
  --models <names of models to be tested, e.g., bert-base-uncased bert-large-uncased> \
  --role_stim datasets/ROLE-88/ROLE-88.tsv \
  --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
  --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
  --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv
```

## Experimental code

`proc_cogstims.py` is the code that was used for the experiments on BERT<sub>BASE</sub> and BERT<sub>LARGE</sub> reported in the paper, including perturbations.

Example usage:
```
python proc_cogstims.py \
  --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv \
  --role_stim datasets/ROLE-88/ROLE-88.tsv \
  --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
  --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
  --resultsdir <location for results files> \
  --bertbase <BERT<sub>BASE</sub> location> \
  --bertlarge <BERT<sub>LARGE</sub> location> \
  --incl_perturb
```

* `bertbase` and `bertlarge` specify locations for PyTorch BERT<sub>BASE</sub> and BERT<sub>LARGE</sub> models -- each folder is expected to include `vocab.txt`, `bert_config.json`, and `pytorch_model.bin` for the corresponding [PyTorch BERT](https://github.com/huggingface/pytorch-transformers) model. (Note that experiments were run with the original pytorch-pretrained-bert version, so I can't guarantee identical results with the updated pytorch-transformers.)
* `incl_perturb` runs experiments with all perturbations reported in the paper. Without this flag, only runs experiments without perturbations.
