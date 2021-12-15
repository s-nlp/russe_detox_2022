# Baselines

We provide two baselines:

- **Delete** -- this is an unsupervised rule-based detoxification model which removes all rude and swear words. The vocabulary of swear words is provided.
- **Fine-tuned T5** -- this is a supervised model which is based on a Russian T5 (Transformer pre-trained on a large number of tasks) which was fine-tuned on the parallel detoxification data which we provide.

