## Unsupervised Baseline: Delete

This is an unsupervised method that eliminates toxic words based on a predefined toxic words vocabulary. The idea is often used on television and other media: rude words are bleeped out or hidden with special characters (usually an asterisk). We provide both the vocabulary and the script that applies it to input sentences.

`toxic_vocab_extended.txt`: the extended version of the original [list](https://github.com/skoltech-nlp/rudetoxifier/blob/main/data/train/MAT_FINAL_with_unigram_inflections.txt) of Russian rude words, extended with their lemmatized versions.
`Delete.ipynb`: the notebook with the method.