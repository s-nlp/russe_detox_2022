import torch
import numpy as np
from tqdm.auto import tqdm, trange


def prepare_target_label(model, target_label):
    if target_label in model.config.id2label:
        pass
    elif target_label in model.config.label2id:
        target_label = model.config.label2id.get(target_label)
    elif target_label.isnumeric() and int(target_label) in model.config.id2label:
        target_label = int(target_label)
    else:
        raise ValueError(f'target_label "{target_label}" is not in model labels or ids: {model.config.id2label}.')
    return target_label


def classify_texts(model, tokenizer, texts, second_texts=None, target_label=None, batch_size=32, verbose=False):
    target_label = prepare_target_label(model, target_label)
    res = []
    if verbose:
        tq = trange
    else:
        tq = range
    for i in tq(0, len(texts), batch_size):
        inputs = [texts[i:i+batch_size]]
        if second_texts is not None:
            inputs.append(second_texts[i:i+batch_size])
        inputs = tokenizer(*inputs, return_tensors='pt', padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            preds = torch.softmax(model(**inputs).logits, -1)[:, target_label].cpu().numpy()
        res.append(preds)
    return np.concatenate(res)


def evaluate_style(
    model,
    tokenizer,
    texts,
    target_label=1,  # 1 is toxic, 0 is neutral
    batch_size=32, 
    verbose=False
):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model,
        tokenizer,
        texts, 
        batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    return rotation_calibration(scores, 0.90)


def evaluate_meaning(
    model,
    tokenizer,
    original_texts, 
    rewritten_texts,
    target_label='entailment', 
    bidirectional=True, 
    batch_size=32, 
    verbose=False, 
    aggregation='prod'
):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model, tokenizer,
        original_texts, rewritten_texts, 
        batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    if bidirectional:
        reverse_scores = classify_texts(
            model, tokenizer,
            rewritten_texts, original_texts,
            batch_size=batch_size, verbose=verbose, target_label=target_label
        )
        if aggregation == 'prod':
            scores = reverse_scores * scores
        elif aggregation == 'mean':
            scores = (reverse_scores + scores) / 2
        elif aggregation == 'f1':
            scores = 2 * reverse_scores * scores / (reverse_scores + scores)
        else:
            raise ValueError('aggregation should be one of "mean", "prod", "f1"')
    return scores


def encode_cls(texts, model, tokenizer, batch_size=32, verbose=False):
    results = []
    if verbose:
        tq = trange
    else:
        tq = range
    for i in tq(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        with torch.no_grad():
            out = model(**tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(model.device))
            embeddings = out.pooler_output
            embeddings = torch.nn.functional.normalize(embeddings).cpu().numpy()
            results.append(embeddings)
    return np.concatenate(results)


def evaluate_cosine_similarity(
    model,
    tokenizer,
    original_texts,
    rewritten_texts,
    batch_size=32,
    verbose=False,
):
    scores = (
        encode_cls(original_texts, model=model, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose)
        * encode_cls(rewritten_texts, model=model, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose)
    ).sum(1)
    return rotation_calibration(scores, 1.50)


def evaluate_cola(
    model,
    tokenizer,
    texts,
    target_label=1,
    batch_size=32, 
    verbose=False
):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model, tokenizer,
        texts, 
        batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    return scores


def evaluate_cola_relative(
    model,
    tokenizer,
    original_texts,
    rewritten_texts,
    target_label=1,
    batch_size=32,
    verbose=False,
    maximum=0,
):
    target_label = prepare_target_label(model, target_label)
    original_scores = classify_texts(
        model, tokenizer,
        original_texts,
        batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    rewritten_scores = classify_texts(
        model, tokenizer,
        rewritten_texts,
        batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    scores = rewritten_scores - original_scores
    if maximum is not None:
        scores = np.minimum(0, scores)
    return rotation_calibration(scores, 1.15, px=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rotation_calibration(data, coef=1.0, px=1, py=1, minimum=0, maximum=1):
    result = (data - px) * coef + py
    if minimum is not None:
        result = np.maximum(minimum, result)
    if maximum is not None:
        result = np.minimum(maximum, result)
    return result


def evaluate_style_transfer(
    original_texts,
    rewritten_texts,
    style_model,
    style_tokenizer,
    meaning_model,
    meaning_tokenizer,
    cola_model,
    cola_tokenizer,
    style_target_label=1,
    batch_size=32,
    verbose=True,
    aggregate=False,
    style_calibration=None,
    meaning_calibration=None,
    fluency_calibration=None,
):
    if verbose: print('Style evaluation')
    accuracy = evaluate_style(
        style_model,
        style_tokenizer,
        rewritten_texts,
        target_label=style_target_label, batch_size=batch_size, verbose=verbose
    )
    if verbose: print('Meaning evaluation')
    similarity = evaluate_cosine_similarity(
        meaning_model,
        meaning_tokenizer,
        original_texts, 
        rewritten_texts,
        batch_size=batch_size, verbose=verbose
    )
    if verbose: print('Fluency evaluation')
    fluency = evaluate_cola_relative(
        cola_model,
        cola_tokenizer,
        rewritten_texts=rewritten_texts,
        original_texts=original_texts,
        batch_size=batch_size, verbose=verbose,
    )

    joint = accuracy * similarity * fluency
    if verbose and (style_calibration or meaning_calibration or fluency_calibration):
        print('Scores:')
        print(f'Style transfer accuracy (STA):  {np.mean(accuracy)}')
        print(f'Meaning preservation (SIM):     {np.mean(similarity)}')
        print(f'Fluency score (FL):             {np.mean(fluency)}')
        print(f'Joint score (J):                {np.mean(joint)}')

    result = dict(
        accuracy=accuracy,
        similarity=similarity,
        fluency=fluency,
        joint=joint
    )
    if aggregate:
        return {k: float(np.mean(v)) for k, v in result.items()}
    return result