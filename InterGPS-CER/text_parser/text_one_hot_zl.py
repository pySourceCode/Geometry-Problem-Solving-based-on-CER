import json
from text_parser import parse
import os
from keras_preprocessing.text import Tokenizer


DATA_PATH = '../geometry3k'
TEXT_OUTPUT_FILE = '../texts'
ONE_HOT_FILE = '../text_one_hot.json'
output_data = {}
texts = []
# one_hot_matrix = []
for pid in range(3002):
    # data splits
    if pid in range(2101):
        split = 'train'
    elif pid in range(2101, 2401):
        split = 'val'
    else:
        split = 'test'

    with open(os.path.join(DATA_PATH, split, str(pid), 'data.json'), 'r') as f:
        data = json.load(f)
    assert str(pid) == str(data['id'])  # prob id: 0-3001
    text = data['compact_text']
    text = text.replace("\\", "")
    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace("\"", "")
    texts.append(text)

os.makedirs(os.path.dirname(TEXT_OUTPUT_FILE), exist_ok=True)
with open(TEXT_OUTPUT_FILE, 'w') as f:
    json.dump(texts, f, indent=2, separators=(',', ': '))

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(texts)
pid = 0
for text in texts:
    logic_forms, output_text, reduced_text = parse(text)
    samples = logic_forms
    # texts 汇总所有问题的文本
    sequences = tokenizer.texts_to_sequences(samples)
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
    word_index = tokenizer.word_index
    # one_hot_matrix[pid][:] = one_hot_results
    # print(one_hot_results)
    output_data[pid] = {'id': pid, 'one_hot': one_hot_results.tolist()}
    pid += 1

train_position = {}

for pid in range(3002):
    train_position[pid] = {'id': pid, 'one_hot': []}
    trainseq = []
    for one_hot in output_data[pid]["one_hot"]:
        trainseq += one_hot
    trainfull = [0 for index in range(1000 - len(trainseq))]
    trainseq += trainfull
    position = []
    for index in range(len(trainseq)):
        if trainseq[index] != 0:
            # print(index)
            position.append(index)
            positionseq = []
    if len(position) <= 10:
        positionfull = [0 for index in range(10 - len(position))]
        positionseq = position + positionfull
    else:
        for i in range(10):
            positionseq.append(position[i])
    train_position[pid]["one_hot"] = positionseq
print(train_position)

os.makedirs(os.path.dirname(ONE_HOT_FILE), exist_ok=True)
with open(ONE_HOT_FILE, 'w') as f:
    json.dump(train_position, f, indent=2, separators=(',', ': '))

