import os
import json


merge_ga_seq = {}
for problem_number in range(2407, 2507):
    merge_ga_seq[problem_number] = {}
    final_path = '..' + str(problem_number) + '.json'
    with open(os.path.join(final_path), 'r') as f:
        data_1 = json.load(f)
    merge_ga_seq[problem_number] = data_1

ga_path = '../merge_GA_seq_2.json'
with open(ga_path, 'w') as f:
    json.dump(merge_ga_seq, f, indent=2, separators=(',', ': '))
