with open("drugprot_training_abstracs.tsv", "r") as f:
    data = f.readlines()
    texts = []
    for line in data:
        texts += line.strip().split('\t')[1:]
with open("train.txt", "w") as f:
    f.writelines('\n'.join(texts))
    
with open("drugprot_development_abstracs.tsv", "r") as f:
    data = f.readlines()
    texts = []
    for line in data:
        texts += line.strip().split('\t')[1:]
with open("dev.txt", "w") as f:
    f.writelines('\n'.join(texts))