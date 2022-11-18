import json
import os

def pre_treatment(DATA_PATH):
    sequence = {}
    with open(os.path.join(DATA_PATH), 'r') as f:
        data = json.load(f)
    for pid in range(2402, 3001):
        sequence[pid] = {'id': pid, 'seq': []}
        for listindex in data[str(pid)]['seq']:
            for index in listindex:
                if index < 0:
                    index = 0
                elif index > 17:
                    index = 17
                else:
                    index = round(index)
                if index != 0:
                    sequence[pid]['seq'].append(index)
    return sequence

if __name__ == "__main__":
    DATA_PATH = '../at_gene_seq_2.json'
    OUTPUT = '../at_seq_pretreat_2.json'
    sequence = pre_treatment(DATA_PATH)
    with open(OUTPUT, 'w') as f:
        json.dump(sequence, f, indent=2, separators=(',', ':'))


