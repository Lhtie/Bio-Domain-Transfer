with open("CDR_TrainingSet.PubTator.txt", "r") as f:
    data = f.readlines()
    texts = []
    for line in data:
        if "|t|" in line or "|a|" in line:
            texts.append(line.strip().split('|')[-1])
with open("train.txt", "w") as f:
    f.writelines('\n'.join(texts))
    
with open("CDR_DevelopmentSet.PubTator.txt", "r") as f:
    data = f.readlines()
    texts = []
    for line in data:
        if "|t|" in line or "|a|" in line:
            texts.append(line.strip().split('|')[-1])
with open("dev.txt", "w") as f:
    f.writelines('\n'.join(texts))