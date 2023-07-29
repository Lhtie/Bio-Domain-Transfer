import openai
import os
import json
import time
from tqdm import tqdm
from tenacity import(
    retry,
    stop_after_attempt,
    wait_random_exponential
)
import evaluate
import numpy as np
from transformers import AutoTokenizer

from dataConfig.chemdner import chemdner

openai.api_key_path = "./openai_api_key"
rpm = 60
data_dir = "/mnt/data/oss_beijing/liuhongyi/datasets/chemdner_corpus"
output_file = "./results/chatgpt_chemdner.json"
template = [
    "Description: In this task, you are given a small paragraph of a PubMed article, and your task is to identify all the named entities (particular chemical related entity) from the given input and also provide type of the each entity according to structure-associated chemical entity mention classes (ABBREVIATION, IDENTIFIER, FORMULA, SYSTEMATIC, MULTIPLE, TRIVIAL, FAMILY). Specifically, the paragraph are given with seperate tokens and you need to list all the chemical named entities in order and also tag their types. Generate the output in this format: entity1 <type_of_entity1>, entity2 <type_of_entity2>.",
    "Examples:",
    "Input: In situ C-C bond cleavage of vicinal diol following by the lactolisation resulted from separated treatment of Arjunolic acid ( 1 ) , 24-hydroxytormentic acid ( 2 ) and 3-O-β-D-glucopyranosylsitosterol ( 3 ) with sodium periodate and silica gel in dried THF according to the strategic position of hydroxyl functions in the molecule .",
    "Output: C-C <FORMULA>, vicinal diol <FAMILY>, Arjunolic acid <TRIVIAL>, 24-hydroxytormentic acid <SYSTEMATIC>, 3-O-β-D-glucopyranosylsitosterol <SYSTEMATIC>, sodium periodate <SYSTEMATIC>, silica gel <TRIVIAL>, THF <ABBREVIATION>, hydroxyl <SYSTEMATIC>",
    "Input: Structural studies using LC/MS/MS analysis and ( 1 ) H NMR spectroscopy showed the formation of a glycosidic bond between the primary hydroxyl group of RVX-208 and glucuronic acid .",
    "Output: ( 1 ) H <FORMULA>, primary hydroxyl <FAMILY>, RVX-208 <IDENTIFIER>, glucuronic acid <TRIVIAL>",
    "Input: The lystabactins are composed of serine ( Ser ) , asparagine ( Asn ) , two formylated/hydroxylated ornithines ( FOHOrn ) , dihydroxy benzoic acid ( Dhb ) , and a very unusual nonproteinogenic amino acid , 4,8-diamino-3-hydroxyoctanoic acid ( LySta ) .",
    "Output: lystabactins <FAMILY>, serine <TRIVIAL>, Ser <FORMULA>, asparagine <TRIVIAL>, Asn <FORMULA>, formylated/hydroxylated ornithines <MULTIPLE>, FOHOrn <ABBREVIATION>, dihydroxy benzoic acid <SYSTEMATIC>, Dhb <ABBREVIATION>, 4,8-diamino-3-hydroxyoctanoic acid <SYSTEMATIC>, LySta <ABBREVIATION>",
    "Please continue:",
    "Input: %s",
    "Output: "
]
# template = [
#     "Given tokens of a paragraph of PubMed article, please tag each chemical named entity token according to structure-associated chemical entity mention classes:",
#     "- ABBREVIATION",
#     "- IDENTIFIER",
#     "- FORMULA",
#     "- SYSTEMATIC",
#     "- MULTIPLE",
#     "- TRIVIAL",
#     "- FAMILY",
#     "Please format the output as list of pairs of tokens and tags (O, B-ABBREVIATION, I-ABBREVIATION, B-IDENTIFIER, I-IDENTIFIER, B-FORMULA, I-FORMULA, B-SYSTEMATIC, I-SYSTEMATIC, B-MULTIPLE, I-MULTIPLE, B-TRIVIAL, I-TRIVIAL, B-FAMILY, I-FAMILY):",
#     "```",
#     "tags = [(token0, tag0), (token1, tag1), ...]",
#     "```",
#     "Paragraph tokens:",
#     "%s"
# ]

# @retry(wait=wait_random_exponential(min=max(1, 60//rpm), max=60), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def load_file():
    preds = {}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            preds = json.load(f)
    return preds

def save_file(preds):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=4, ensure_ascii=False)

def convert_ans(content, response):
    tags = []
    pt = 0
    for line in response.split('\n'):
        if ">, " in line:
            for ctx in line.split(">, "):
                if " <" in ctx:
                    token, tag = ctx.split(" <")[:2]
                    token = token.split(' ')
                    tag = tag.lstrip('<').rstrip('>')
                    while pt < len(content) and content[pt:pt+len(token)] != token:
                        tags.append('O')
                        pt += 1
                    if pt < len(content):
                        tags += ["B-" + tag] + ["I-" + tag] * (len(token) - 1)
                        pt += len(token)
    while pt < len(content):
        tags.append('O')
        pt += 1
    return tags

def convert_ans_fillall(content, response):
    tags = ['O'] * len(content)
    for line in response.split('\n'):
        if ">, " in line:
            for ctx in line.split(">, "):
                if " <" in ctx:
                    token, tag = ctx.split(" <")[:2]
                    token = token.split(' ')
                    tag = tag.lstrip('<').rstrip('>')
                    for pt in range(0, len(content) - len(token) + 1):
                        if content[pt:pt+len(token)] == token:
                            tags[pt:pt+len(token)] = ["B-" + tag] + ["I-" + tag] * (len(token) - 1)
    return tags

def requestAPI(idx, content, preds):
    prompts = [
        {
            "role": "system",
            "content": "You are an expert of chemical named entity recognition tasks"
        },
        {
            "role": "user",
            "content": '\n'.join(template) % ' '.join(content)
        }
    ]
    response = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=prompts
    )['choices'][0]['message']['content']

    preds[str(idx)] = [response, convert_ans(content, response)]
    save_file(preds)

def trying(dataset, test_ids, preds):
    for idx in tqdm(test_ids):
        if str(idx) in preds:
            continue
        input_tokens = dataset[idx]["tokens"]
        requestAPI(idx, input_tokens, preds)
        time.sleep(80 / rpm)

    dead = []
    for idx in test_ids:
        if not str(idx) in preds:
            dead.append(idx)
    return dead

def chatgpt_chemdner(dataset):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    num_test = len(dataset)
    preds = load_file()
    remaining_ids = []
    for idx in range(num_test):
        if not str(idx) in preds:
            remaining_ids.append(idx)
    if len(remaining_ids) == 0:
        print(f"Completed, skip")
        return
    if len(remaining_ids) < num_test:
        print(f"Detect {num_test - len(remaining_ids)} completed, test the rest {len(remaining_ids)} only...")

    while True:
        remaining_ids = trying(dataset, remaining_ids, preds)
        if len(remaining_ids) > 0:
            print(f"remaining ids: {remaining_ids}")
            print("retrying all remaining ids...")
        else:
            print(f"All completed")
            break

def evalution(dataConfig, dataset):
    preds = load_file()
    predictions = []
    for key, pred in preds.items():
        predictions.append([dataConfig.label2id.get(label, 0) for label in pred[1]])
    references = [batch['ner_tags'] for batch in dataset]

    assert len(predictions) == len(references), f"{len(predictions)} != {len(references)}"
    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        assert len(pred) == len(ref), f"{idx}: {len(pred)} != {len(ref)}"
        predictions[idx] = [dataConfig.id2label[p] for p in pred]
        references[idx] = [dataConfig.id2label[r] for r in ref]

    seqeval = evaluate.load('evaluate-metric/seqeval')
    results = seqeval.compute(predictions=predictions, references=references)
    print(results)

if __name__ == '__main__':
    dataConfig = chemdner(data_dir)
    dataset = dataConfig.load_dataset()['evaluation']
    chatgpt_chemdner(dataset)

    # preds = load_file()
    # for idx, batch in enumerate(dataset):
    #     pred = convert_ans_fillall(batch['tokens'], preds[str(idx)][0])
    #     preds[str(idx)][1] = pred
    # save_file(preds)
    
    evalution(dataConfig, dataset)