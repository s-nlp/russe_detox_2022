# The First Competition on Detoxification for Russian

This repository contains the data and scripts for the [Detoxification shared task](https://russe.nlpub.org/2022/tox/) at Dialogue-2022. You can participate in the shared task by submitting your models to [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/642).

## Baselines

We provide two baselines:
- **Delete** -- this is an unsupervised rule-based detoxification model which removes all rude and swear words. The vocabulary of swear words is provided.
- **Fine-tuned T5** -- this is a supervised model which is based on a Russian [T5](https://arxiv.org/abs/1910.10683) (Transformer pre-trained on a large number of tasks) which was fine-tuned on the parallel detoxification data which we provide.

## Data

We provide a parallel detoxification dataset: Russian toxic sentences and their detoxified version which were manually written and validated by crowd workers:
- **training** (!!!updated 29.12.2021!!!) - 6,948 sentences with 1 to 3 detoxified versions.
- **development** - 800 sentences with 1 to 3 detoxified versions.

Test set will be made available during the evaluation phase.

## Evaluation

We provide scripts which will be used for automatic evaluation of the models during the development phase of the competition. These are the same versions as the ones we use at CodaLab, so you should get the same scores as there.

We compute the following metrics:
- **Style transfer accuracy (STA)** -- the average confidence of the pre-trained BERT-based toxicity classifier for the output sentences.
- **Meaning preservation (SIM)** -- the distance of embeddings of the input and output sentences. The embeddings are generated with the [LaBSE](https://arxiv.org/abs/2007.01852) model.
- **Fluency score (FL)** -- the average confidence of the BERT-based fluency classifier trained to discriminate between real and corrupted sentences.
- **Joint score (J)** -- the sentence-level multiplication of the STA, SIM, and FL scores.
- **chrF** -- the chrF metric computed with respect to reference texts.
