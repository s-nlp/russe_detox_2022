# RuParaDetox: The First Competition on Detoxification for Russian

This repository contains the data and scripts for the [Text Detoxification shared task RUSSE2022](https://russe.nlpub.org/2022/tox/) at Dialogue-2022. You can participate in the shared task by submitting your models to [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/642).

📰 **Updates**

Check out **TextDetox** 🤗 https://huggingface.co/collections/textdetox/ -- continuation of ParaDetox project!

**[2025] !!!NOW OPEN!!! TextDetox CLEF2025 shared task: for even more -- 15 languages!** [website](https://pan.webis.de/clef25/pan25-web/text-detoxification.html) 🤗[Starter Kit](https://huggingface.co/collections/textdetox/)

**[2025] COLNG2025**: Daryna Dementieva, Nikolay Babakov, Amit Ronen, Abinew Ali Ayele, Naquee Rizwan, Florian Schneider, Xintong Wang, Seid Muhie Yimam, Daniil Alekhseevich Moskovskiy, Elisei Stakovskii, Eran Kaufman, Ashraf Elnagar, Animesh Mukherjee, and Alexander Panchenko. 2025. ***Multilingual and Explainable Text Detoxification with Parallel Corpora***. In Proceedings of the 31st International Conference on Computational Linguistics, pages 7998–8025, Abu Dhabi, UAE. Association for Computational Linguistics. [pdf](https://aclanthology.org/2025.coling-main.535/)

**[2024]** We have also created versions of ParaDetox in more languages. You can checkout a [RuParaDetox](https://huggingface.co/datasets/s-nlp/ru_paradetox) dataset as well as a [Multilingual TextDetox](https://huggingface.co/textdetox) project that includes 9 languages.

Corresponding papers:
* [MultiParaDetox: Extending Text Detoxification with Parallel Data to New Languages](https://aclanthology.org/2024.naacl-short.12/) (NAACL 2024)
* [Overview of the multilingual text detoxification task at pan 2024](https://ceur-ws.org/Vol-3740/paper-223.pdf) (CLEF Shared Task 2024)

## Baselines

We provide two baselines:
- **Delete** -- this is an unsupervised rule-based detoxification model which removes all rude and swear words. The vocabulary of swear words is provided.
- **Fine-tuned T5** -- this is a supervised model which is based on a Russian [T5](https://arxiv.org/abs/1910.10683) (Transformer pre-trained on a large number of tasks) which was fine-tuned on the parallel detoxification data which we provide. 🤗[ruT5-base-detox](https://huggingface.co/s-nlp/ruT5-base-detox)

## Data

We provide a parallel detoxification dataset: Russian toxic sentences and their detoxified version which were manually written and validated by crowd workers:
- **training** (!!!updated 29.12.2021!!!) - 6,948 sentences with 1 to 3 detoxified versions.
- **development** - 800 sentences with 1 to 3 detoxified versions.

Test set will be made available during the evaluation phase.

The data is available online: 🤗[ru-paradetox](https://huggingface.co/datasets/s-nlp/ru_paradetox)

### Annotation Steps Data

We also release publically 🤗 the results of data collection from each crowdsourcing annotation task:
* *Task 1: Generation of Paraphrases*: [s-nlp/ru_non_detoxified](https://huggingface.co/datasets/s-nlp/ru_non_detoxified)
* *Task 2: Content Preservation Check*: [s-nlp/ru_paradetox_content](https://huggingface.co/datasets/s-nlp/ru_paradetox_content)
* *Task 3: Toxicity Check*: [s-nlp/ru_paradetox_toxicity](https://huggingface.co/datasets/s-nlp/ru_paradetox_toxicity)

## Evaluation

We provide scripts which will be used for automatic evaluation of the models during the development phase of the competition. These are the same versions as the ones we use at CodaLab, so you should get the same scores as there.

We compute the following metrics:
- **Style transfer accuracy (STA)** -- the average confidence of the pre-trained BERT-based toxicity classifier for the output sentences. 🤗 [Classifier](https://huggingface.co/s-nlp/russian_toxicity_classifier)
- **Meaning preservation (SIM)** -- the distance of embeddings of the input and output sentences. The embeddings are generated with the [LaBSE](https://arxiv.org/abs/2007.01852) model.
- **Fluency score (FL)** -- the average confidence of the BERT-based fluency classifier trained to discriminate between real and corrupted sentences.
- **Joint score (J)** -- the sentence-level multiplication of the STA, SIM, and FL scores.
- **chrF** -- the chrF metric computed with respect to reference texts.

## Citation

```
@article{Dementieva2022RUSSE2022,
	title        = {{RUSSE-2022: Findings of the First Russian Detoxification Shared Task Based on Parallel Corpora}},
	author       = {Daryna Dementieva and Varvara Logacheva and Irina Nikishina and Alena Fenogenova and David Dale and I. Krotova and Nikita Semenov and Tatiana Shavrina and Alexander Panchenko},
	year         = 2022,
	journal      = {COMPUTATIONAL LINGUISTICS AND INTELLECTUAL TECHNOLOGIES},
	url          = {https://api.semanticscholar.org/CorpusID:253169495}
}
```

```
@inproceedings{dementieva-etal-2024-multiparadetox,
    title = "{M}ulti{P}ara{D}etox: Extending Text Detoxification with Parallel Data to New Languages",
    author = "Dementieva, Daryna  and
      Babakov, Nikolay  and
      Panchenko, Alexander",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-short.12",
    pages = "124--140",
    abstract = "Text detoxification is a textual style transfer (TST) task where a text is paraphrased from a toxic surface form, e.g. featuring rude words, to the neutral register. Recently, text detoxification methods found their applications in various task such as detoxification of Large Language Models (LLMs) (Leong et al., 2023; He et al., 2024; Tang et al., 2023) and toxic speech combating in social networks (Deng et al., 2023; Mun et al., 2023; Agarwal et al., 2023). All these applications are extremely important to ensure safe communication in modern digital worlds. However, the previous approaches for parallel text detoxification corpora collection{---}ParaDetox (Logacheva et al., 2022) and APPADIA (Atwell et al., 2022){---}were explored only in monolingual setup. In this work, we aim to extend ParaDetox pipeline to multiple languages presenting MultiParaDetox to automate parallel detoxification corpus collection for potentially any language. Then, we experiment with different text detoxification models{---}from unsupervised baselines to LLMs and fine-tuned models on the presented parallel corpora{---}showing the great benefit of parallel corpus presence to obtain state-of-the-art text detoxification models for any language.",
}
```

## Contact

For any questions, test data request, please, contact: [Daryna Dementieva](mailto:dardem96@gmail.com)
