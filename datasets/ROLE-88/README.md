ROLE-88 items come from Chow et al. (2015). The authors of this study have also kindly provided the cloze data for these items, allowing us to include them here.

The `ROLE-88.tsv` file contains for each item the item ID, the context, the expected item for that context, the cloze value of the expected item, the target, the cloze value of the target, and the strict cloze value of the target.

* The expected word is the word (or words) that received the highest cloze probability for the context.
* The target word is the word used for the sensitivity tests. (This is the only dataset for which there is a separate "target" category used for sensitivity tests. Usually the target is the same as one of the expected items, as in the other datasets, but for a few items they are different.)
* "tgt_cloze(strict)" does not count passive forms of a verb in the cloze probability for the active verb as (e.g., 'been served by' does not count toward 'served'). "tgt_cloze(strict)" does count passives toward the active form.

**Reference**:
* Wing-Yee Chow, Cybelle Smith, Ellen Lau, and Colin Phillips. 2016. *A ‘bag-of-arguments’ mechanism for initial verb predictions.*
