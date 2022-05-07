This dataset contains two subsets: NEG-136-SIMP and NEG-136-NAT. NEG-136-SIMP items come from Fischler et al. (1983). NEG-136-NAT items come from Nieuwland & Kuperberg (2008).

The `NEG-136-SIMP.tsv` and `NEG-136-NAT.tsv` files contain for each item the affirmative and negative version of the context (context_aff, context_neg), and completions that are true with the affirmative context (target_aff) and with the negative context (target_neg).

* For NEG-136-SIMP, determiners (*a*/*an*) are left ambiguous, and need to be selected based on the completion noun (this is done already in `proc_datasets.py`).

**References**:
* Ira Fischler, Paul A Bloom, Donald G Childers, Salim E Roucos, and Nathan W Perry Jr. 1983. *Brain potentials related to stages of sentence verification.*
* Mante S Nieuwland and Gina R Kuperberg. 2008. *When the truth is not too hard to handle: An event-related potential study on the pragmatics of negation.*
