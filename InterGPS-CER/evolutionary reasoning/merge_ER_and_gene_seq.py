import os
import json

ga_path = '../merge_GA_seq_2.json'
gene_path = '../at_seq_pretreat_1.json'
gene_path2 = '../at_seq_pretreat_2.json'
with open(os.path.join(ga_path), 'r') as f:
    data_1 = json.load(f)
with open(os.path.join(gene_path), 'r') as f:
    data_2 = json.load(f)
with open(os.path.join(gene_path2), 'r') as f:
    data_3 = json.load(f)

merge_ga_and_gene_seq = {}
for problem_number in range(2407, 2507):
    merge_ga_and_gene_seq[problem_number] = {'id':problem_number, 'seq': []}

    ga_seq = data_1[str(problem_number)]['seq']
    gene_seq1 = data_2[str(problem_number)]['seq']
    gene_seq2 = data_3[str(problem_number)]['seq']

    merge_ga_and_gene_seq[problem_number]['seq'].append(ga_seq)
    merge_ga_and_gene_seq[problem_number]['seq'].append(gene_seq1)
    merge_ga_and_gene_seq[problem_number]['seq'].append(gene_seq2)

final_path = '../merge_GA_and_gene_seq_2.json'
with open(final_path, 'w') as f:
    json.dump(merge_ga_and_gene_seq, f, indent=2, separators=(',', ': '))
