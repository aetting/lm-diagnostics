CPRAG-34 items come from Federmeier and Kutas (1999).

The `CPRAG-34.tsv` file contains for each item the item ID, the context, three completion words, and the constraint category according to the original Federmeier & Kutas paper.

* "expected" completions are the highest cloze.
* "within_category" completions are approximately zero cloze but share a close semantic category with the expected word (e.g., sports).
* "between-category" completions are approximately zero cloze and do not share the immediate semantic category with the expected word, but share a broader semantic category (e.g., games).
* "constraint" indicates High- or Low-constraint context, based on the cloze of the expected word, binned by a median split on the expected target cloze values across items in the original experiment.

The `CPRAG-explanations.tsv` file contains explanations of the commonsense/pragmatic inference involved in predicting the correct words in this set.

**Reference**:
* Kara D Federmeier and Marta Kutas. 1999. *A rose by any other name: Long-term memory structure and sentence processing.*
