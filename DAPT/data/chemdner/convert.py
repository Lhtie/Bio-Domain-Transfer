with open("training.abstracts.txt", "r") as f:
    data = f.readlines()
    texts = []
    for line in data:
        texts += line.strip().split('\t')[1:]
with open("train.txt", "w") as f:
    f.writelines('\n'.join(texts))
    
with open("development.abstracts.txt", "r") as f:
    data = f.readlines()
    texts = []
    for line in data:
        texts += line.strip().split('\t')[1:]
with open("dev.txt", "w") as f:
    f.writelines('\n'.join(texts))