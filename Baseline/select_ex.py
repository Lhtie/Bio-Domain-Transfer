from dataConfig.chemdner import chemdner
from datasets import Dataset

chemdner = chemdner('Try', granularity='sent', oracle=True)
dataset = chemdner.load_dataset()
label_list = [ 
    "O", "B-ABBREVIATION", "I-ABBREVIATION", "B-IDENTIFIER", "I-IDENTIFIER", "B-FORMULA", "I-FORMULA", "B-SYSTEMATIC", "I-SYSTEMATIC", "B-MULTIPLE", "I-MULTIPLE", "B-TRIVIAL", "I-TRIVIAL", "B-FAMILY", "I-FAMILY"
]

for key in ['training']:
    num = len(dataset[key])
    cnt = []
    for idx, tags in enumerate(dataset[key]['ner_tags']):
        sets = set(tags)
        cnt.append(len(sets))
    indices = sorted(range(len(cnt)), key=lambda x: cnt[x], reverse=True)
    
    dataset[key] = Dataset.from_dict(dataset[key][indices[:20]])

    label_cnt = {}
    for labels in dataset[key]['ner_tags']:
        for label in labels:
            if not label in label_cnt:
                label_cnt[label] = 0
            label_cnt[label] += 1
    print(label_cnt)

    for batch in dataset[key]:
        print(' '.join(batch['tokens']))
        print(', '.join([
            f"{token} <{label}>" for token, label in zip(batch['tokens'], [label_list[tag] for tag in batch['ner_tags']])
        ]))
        print()
