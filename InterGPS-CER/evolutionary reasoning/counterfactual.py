import os
import json
import numpy as np


def expert_knowledge():
    expert_path1 = '../pred_seqs_train_merged_correct.json'
    with open(os.path.join(expert_path1), 'r') as f:
        data_1 = json.load(f)

    frontrelation_mat = [[0] * 17 for _ in range(17)]
    behindrelation_mat = [[0] * 17 for _ in range(17)]
    for pid, res in data_1.items():
        for index in data_1[pid]['seqs']:
            max_len = len(index)
            if max_len == 1:
                continue
            for element in range(1, max_len):
                frontrelation_mat[index[element] - 1][index[element - 1] - 1] += 1
            for element in range(0, max_len - 1):
                behindrelation_mat[index[element] - 1][index[element + 1] - 1] += 1

    return frontrelation_mat, behindrelation_mat


def list_calcu(lst):
    max1 = max(lst)
    max_pos1 = lst.index(max1)
    temp = lst
    temp[max_pos1] = 0
    max2 = max(temp)
    max_pos2 = temp.index(max2)
    temp[max_pos1] = max1
    p1 = max1/(max1+max2)
    p2 = max2/(max1+max2)
    return max_pos1, max_pos2, p1, p2


def counterfactual_variation(number):
    frontrelation_mat, behindrelation_mat = expert_knowledge()
    # print(frontrelation_mat)
    # print(behindrelation_mat)
    row_max_pos1, row_max_pos2, p1, p2 = list_calcu(frontrelation_mat[number - 1])
    # print(row_max_pos1, row_max_pos2,p1, p2)
    colomn_max_pos11, colomn_max_pos12, p11, p12 = list_calcu(behindrelation_mat[:][row_max_pos1])
    # print(colomn_max_pos11,colomn_max_pos12,p11,p12)
    colomn_max_pos21, colomn_max_pos22, p21, p22 = list_calcu(behindrelation_mat[:][row_max_pos2])
    # print(colomn_max_pos21, colomn_max_pos22, p21, p22)

    if max(p1*p11, p1*p12, p2*p21) == p1*p11:
        final_p1 = p1*p11
        row_pos1 = row_max_pos1
        colomn_pos1 = colomn_max_pos11
        if p1*p12 > p2*p21:
            final_p2 = p1*p12
            row_pos2 = row_max_pos1
            colomn_pos2 = colomn_max_pos12
        else:
            final_p2 = p2*p21
            row_pos2 = row_max_pos2
            colomn_pos2 = colomn_max_pos21
    elif max(p1*p11, p1*p12, p2*p21) == p1*p12:
        final_p1 = p1*p12
        row_pos1 = row_max_pos1
        colomn_pos1 = colomn_max_pos12
        if p1*p11 > p2*p21:
            final_p2 = p1*p11
            row_pos2 = row_max_pos1
            colomn_pos2 = colomn_max_pos11
        else:
            final_p2 = p2*p21
            row_pos2 = row_max_pos2
            colomn_pos2 = colomn_max_pos21
    else:
        final_p1 = p2*p21
        row_pos1 = row_max_pos2
        colomn_pos1 = colomn_max_pos21
        if p1*p11 > p1*p12:
            final_p2 = p1*p11
            row_pos2 = row_max_pos1
            colomn_pos2 = colomn_max_pos11
        else:
            final_p2 = p1*p12
            row_pos2 = row_max_pos1
            colomn_pos2 = colomn_max_pos12

    # print(row_pos1, colomn_pos1)
    # print(row_pos2, colomn_pos2)
    changed_p1 = 0.8 * final_p1/(final_p1+final_p2)
    changed_p2 = 0.8-changed_p1
    x = np.random.rand()
    if x < changed_p1:
        # print("situation:1")
        variation = colomn_pos1+1
    elif changed_p1 <= x < 0.8:
        # print("situation:2")
        variation = colomn_pos2+1
    else:
        # print("situation:3")
        variation = np.random.randint(1, 18)

    # print(variation)
    # print(behindrelation_mat)
    # print(behindrelation_mat[9][16])
    return variation



