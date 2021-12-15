## Evaluation

This folder contains the copy of the code for metrics that is used in Codalab. For your convenience, we provide both notebook and script format of metrics launch.

You can launch all metrics one by one in the notebook `ru_detoxification_evaluation.ipynb`.

Also, you can launch all evaluation pipeline via `ru_detoxification_evaluation.py` script with the following parameters:
- `-i`: the path to the input dataset in `.tsv` format;
- `-p`: the path to the file of model's prediction written in `.txt` file;
- `-r`: if there is a need to calculate ChrF score with references;
- `-n`: the name of your model;
- `--batch_size`: size of batch for metric's models inference;
- `--use_cuda`: if the usage of `cuda` if possible, if not the inference will be done on `cpu`.

The example run:

`python ru_detoxification_evaluation.py -i ../data/input/dev.tsv -p ../data/output/t5_base_10000_dev.txt -r True -n t5`

The script will create `results.md` file and will update it if you run new experiments.
