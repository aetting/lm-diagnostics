# LM diagnostics

This repository contains the diagnostic datasets and experimental code for 'What BERT is not'.

# Diagnostic test data

The `datasets` folder contains TSV files with data for each diagnostic test, along with explanatory README files for each dataset.

# Experimental code

The code provided was used for reading in datasets, retrieving word predictions and probabilities from BERT<sub>BASE</sub> and BERT<sub>LARGE</sub>, and performing the relevant calculations for the experiments reported in the paper. `run_aux_tests` also runs the experiments with the various perturbations.

Example usage
```
python proc_cogstims.py \
  --cprag_stim $bertdir/features/FKstims-BV-clean.tsv \
  --role_stim $bertdir/features/WY-BV-clean.tsv \
  --negnat_stim $bertdir/features/NK_stims-clean.tsv \
  --negsimp_stim $bertdir/features/fischler_stims-clean.tsv \
  --resultsdir $bertdir/results \
  --bertbase $bertdir/params/bert-base-uncased \
  --bertlarge $bertdir/params/bert-large-uncased

```

* `resultsdir` is the location results files and log files will write to
* `bertbase` and `bertlarge` is the location of pytorch parameters and vocab file for BERT<sub>BASE</sub> and BERT<sub>LARGE</sub> models
