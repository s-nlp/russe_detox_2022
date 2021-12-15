import os
import argparse
import torch
from ru_detoxification_metrics import evaluate_style_transfer, rotation_calibration
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from nltk.translate.chrf_score import corpus_chrf
import pandas as pd


def load_model(model_name=None, model=None, tokenizer=None,
               model_class=AutoModelForSequenceClassification, use_cuda=True):
    if model is None:
        if model_name is None:
            raise ValueError('Either model or model_name should be provided')
        model = model_class.from_pretrained(model_name)
        if torch.cuda.is_available() and use_cuda:
            model.cuda()
    if tokenizer is None:
        if model_name is None:
            raise ValueError('Either tokenizer or model_name should be provided')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def evaluate(original, rewritten, batch_size):
    return evaluate_style_transfer(
        original_texts=original,
        rewritten_texts=rewritten,
        style_model=style_model,
        style_tokenizer=style_tokenizer,
        meaning_model=meaning_model,
        meaning_tokenizer=meaning_tokenizer,
        cola_model=fluency_model,
        cola_tokenizer=fluency_tolenizer,
        style_target_label=0,
        batch_size=batch_size,
        aggregate=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputs", help="path to test sentences", required=True)
    parser.add_argument('-p', "--preds", help="path to predictions of a model", required=True)
    parser.add_argument('-r', "--references", help="if calcualte ChrF1 with references", default=False, type=bool)
    parser.add_argument('-n', "--name", help="model name", default='test', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--use_cuda", default=False, type=bool)

    args = parser.parse_args()

    df = pd.read_csv(args.inputs, sep='\t')
    df = df.fillna('')
    toxic_inputs = df['toxic_comment'].tolist()

    with open(args.preds, 'r') as preds_file:
        rewritten = preds_file.readlines()
        rewritten = [sentence.strip() for sentence in rewritten]

    print('Models loading...')
    style_model, style_tokenizer = load_model('SkolkovoInstitute/russian_toxicity_classifier', use_cuda=args.use_cuda)
    meaning_model, meaning_tokenizer = load_model('cointegrated/LaBSE-en-ru', use_cuda=args.use_cuda,
                                                  model_class=AutoModel)
    fluency_model, fluency_tolenizer = load_model('SkolkovoInstitute/rubert-base-corruption-detector',
                                                  use_cuda=args.use_cuda)

    results = evaluate(toxic_inputs, rewritten, args.batch_size)

    if not os.path.exists('results.md'):
        with open('results.md', 'w') as f:
            f.writelines('| Model | ACC | SIM | FL | J | ChrF1 |\n')
            f.writelines('| ----- | --- | --- | -- | - | ---- |\n')

    if args.references:
        neutral_references = []
        for index, row in df.iterrows():
            neutral_references.append([row['neutral_comment1'], row['neutral_comment2'], row['neutral_comment3']])

        chrf = corpus_chrf(neutral_references, rewritten)
    else:
        chrf = float('nan')

    with open('results.md', 'a') as res_file:
        res_file.writelines(f"{args.name}|{results['accuracy']:.4f}|{results['similarity']:.4f}|"
                            f"{results['fluency']:.4f}|{results['joint']:.4f}|{chrf:.4f}|\n")