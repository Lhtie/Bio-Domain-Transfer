import json
import os
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer

import sys
sys.path.append("../")
from dataConfig.chemdner import chemdner

results_dir = "../results"
id2labels = [ 
    "O", "B-ABBREVIATION", "I-ABBREVIATION", "B-IDENTIFIER", "I-IDENTIFIER", "B-FORMULA", "I-FORMULA", "B-SYSTEMATIC", "I-SYSTEMATIC", "B-MULTIPLE", "I-MULTIPLE", "B-TRIVIAL", "I-TRIVIAL", "B-FAMILY", "I-FAMILY"
]

# copy from https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py

def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

def extract_tp_actual_correct(y_true, y_pred, suffix=False, *args):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    for type_name, start, end in get_entities(y_true, suffix):
        entities_true[type_name].add((start, end))
    for type_name, start, end in get_entities(y_pred, suffix):
        entities_pred[type_name].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum

if __name__ == "__main__":
    # model_name = "bert-base-uncased"
    model_names = [
        "bert-base-uncased",
        "bert-base-uncased",
        "bert-base-uncased",
        "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        # "bert-base-uncased",
        # "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    ]
    result_files = [
        "baseline/chemdner-chemdner_ner_bert-base-uncased.json",
        "baseline/chemdner-pc_ner_bert-base-uncased_2stage.json",
        "chemdner-pc_ner_bert-base-uncased_2stage.json",
        "baseline/chemdner-chemdner_ner_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext.json",
        # "chemdner_oracle-chemdner_oracle_ner_bert-base-uncased.json",
        # "chemdner_oracle-chemdner_oracle_ner_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext.json"
    ]

    tps, fps, fns = [], [], []
    for model_name, result_file in zip(model_names, result_files):
        file_path = os.path.join(results_dir, result_file)

        with open(file_path, "r") as f:
            d = json.load(f)
            preds = d["predictions"]
            refs = d["references"]
        
        cnt, tot, pred_tot, ref_tot = 0, 0, 0, 0
        pred_sum, tp_sum, true_sum = extract_tp_actual_correct(refs, preds)

        tps.append(tp_sum.sum())
        fps.append(pred_sum.sum() - tp_sum.sum())
        fns.append(true_sum.sum() - tp_sum.sum())
        print(true_sum)

    print('\t'.join([str(x) for x in tps]))
    print('\t'.join([str(x) for x in fps]))
    print('\t'.join([str(x) for x in fns]))
    
    if True:
        model_names = [
            "bert-base-uncased",
        ]
        result_files = [
            "baseline/chemdner-chemdner_ner_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext.json",
        ]   
        chemdner = chemdner(model_name, cache_dir="../.cache/")
        for model_name, result_file in zip(model_names, result_files):
            tokenizer = AutoTokenizer.from_pretrained(f"/mnt/data/oss_beijing/liuhongyi/models/{model_name}")
            dataset = chemdner.load(tokenizer)['evaluation']

            file_path = os.path.join(results_dir, result_file)
            with open(file_path, "r") as f:
                d = json.load(f)
                preds = d["predictions"]
                refs = d["references"]

            fp, true_cover = [], []
            for tokens, pred, ref in zip(dataset["tokens"], preds, refs):
                entities_true = set()
                entities_pred = set()
                for type_name, start, end in get_entities(ref):
                    entities_true.add((type_name, start, end))
                for type_name, start, end in get_entities(pred):
                    entities_pred.add((type_name, start, end))
                for type_name, start, end in entities_pred - entities_true:
                    fp.append(' '.join(tokens[start:end+1]))

                entities_true = set()
                entities_pred = set()
                for type_name, start, end in get_entities(ref):
                    entities_true.add((start, end))
                for type_name, start, end in get_entities(pred):
                    entities_pred.add((start, end))
                for start, end in entities_pred - entities_true:
                    true_cover.append(' '.join(tokens[start:end+1]))
            
            print(len(true_cover), len(fp) - len(true_cover))
                    
            with open("/mnt/data/oss_beijing/liuhongyi/datasets/chemdner_corpus/evaluation.annotations.txt", "r") as f:
                annotation = f.read()
            sieved_fp = []
            for f in fp:
                if not f in annotation:
                    sieved_fp.append(f)
            with open("fp_tokens.json", "w", encoding="utf-8") as f:
                json.dump(sieved_fp, f, indent=4, ensure_ascii=False)
            
            # fn = []
            # for tokens, pred, ref in zip(dataset["tokens"], preds, refs):
            #     entities_true = set()
            #     entities_pred = set()
            #     for type_name, start, end in get_entities(ref):
            #         entities_true.add((type_name, start, end))
            #     for type_name, start, end in get_entities(pred):
            #         entities_pred.add((type_name, start, end))
            #     for type_name, start, end in entities_true - entities_pred:
            #         fn.append(' '.join(tokens[start:end+1]))
                    
            # with open("fn_tokens.json", "w", encoding="utf-8") as f:
            #     json.dump(fn, f, indent=4, ensure_ascii=False)