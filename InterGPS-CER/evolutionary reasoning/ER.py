import sys
sys.path.append("..")
sys.path.append("../symbolic_solver")
import time

import numpy as np
import os
import json
import argparse
# from func_timeout import func_timeout, FunctionTimedOut
from symbolic_solver.extended_definition import ExtendedDefinition
from symbolic_solver.logic_parser import ExtendedDefinition
from symbolic_solver.logic_parser import LogicParser
from symbolic_solver.logic_solver_with_usage import LogicSolver_v2
from counterfactual import counterfactual_variation


def get_parameters():  # 需要根据调用情况进行修改
    parser = argparse.ArgumentParser(description="Welcome to use GeoSolver!")

    # experiment label and strategy
    # parser.add_argument("--label", type=str, required=True, help="the label of current experiment")
    parser.add_argument("--strategy", type=str, default="predict", help="different search strategies")

    # input data
    parser.add_argument("--data_path", type=str, default="../data/geometry3k", help="the path of geometry3k")
    parser.add_argument("--text_logic_form_path", type=str, help="the path of text logic forms")
    parser.add_argument("--diagram_logic_form_path", type=str, help="the path of diagram logic forms")

    # important parameters for the symbolic solver
    parser.add_argument("--use_annotated", action="store_true", help="use annotated data instead of generated data")
    parser.add_argument("--predict_path", type=str, help="the predict sequence for the solver")
    parser.add_argument("--start_index", type=int, default=2401, help="the start point of testing data")
    parser.add_argument("--end_index", type=int, default=3001, help="the end point of testing data")
    parser.add_argument("--time_limit", type=int, default=150, help="the seconds of time limit")
    parser.add_argument("--num_threads", type=int, default=20,
                        help="the number of running threads, recommendation: # of CPU threads")

    args = parser.parse_args()

    # text logic forms
    if args.text_logic_form_path is None:
        if args.use_annotated:
            args.text_logic_form_path = "../data/geometry3k/logic_forms/" \
                                        "text_logic_forms_annot_dissolved.json"
        else:
            args.text_logic_form_path = "../text_parser/text_logic_forms_pred.json"

    # diagram logic forms
    if args.diagram_logic_form_path is None:
        if args.use_annotated:
            args.diagram_logic_form_path = "../data/geometry3k/logic_forms/" \
                                           "diagram_logic_forms_annot.json"
        else:
            args.diagram_logic_form_path = "../diagram_parser/diagram_logic_forms_pred.json"

    # args.low_first = args.strategy in ["low-first", "final"]  # apply the low-first search strategy
    args.step_limit = 100  # the maximum search steps
    args.debug_mode = False  # debug mode
    args.enable_round = False
    args.enable_predict = True
    args.low_first = False

    return args


def get_forms():
    args = get_parameters()
    para_lst = {}

    text_logic_table = json.load(open(args.text_logic_form_path, "r"))

    diagram_logic_table = None
    if args.diagram_logic_form_path is not None:
        diagram_logic_table = json.load(open(args.diagram_logic_form_path, "r"))

    lst = list(range(2403, 3002))
    for index in lst:
        para_lst[index] = {}
        str_index = str(index)
        text_logic_form, diagram_logic_form = None, None

        if text_logic_table is not None:
            text_logic_form = text_logic_table.get(str_index)

        if diagram_logic_table is not None:
            diagram_logic_form = diagram_logic_table.get(str_index)

        para_lst[index]['args'] = args
        para_lst[index]['text_logic_forms'] = text_logic_form
        para_lst[index]['diagram_logic_forms'] = diagram_logic_form

    return para_lst


def solve_problem(args, text_parser, diagram_parser, order_lst):

    parser = LogicParser(ExtendedDefinition(debug=args.debug_mode))

    if diagram_parser is not None:
        # Define diagram primitive elements
        parser.logic.point_positions = diagram_parser['point_positions']

        isLetter = lambda ch: ch.upper() and len(ch) == 1
        parser.logic.define_point([_ for _ in parser.logic.point_positions if isLetter(_)])
        # if args.debug_mode:
            # print()
            # print(parser.logic.point_positions)
        lines = diagram_parser['line_instances']  # ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
        for line in lines:
            line = line.strip()
            if len(line) == 2 and isLetter(line[0]) and isLetter(line[1]):
                parser.logic.define_line(line[0], line[1])

        circles = diagram_parser['circle_instances']  # ['O']
        for point in circles:
            parser.logic.define_circle(point)

        # Parse diagram logic forms
        logic_forms = diagram_parser['diagram_logic_forms']
        logic_forms = sorted(logic_forms, key=lambda x: x.find("Perpendicular") != -1)  # put 'Perpendicular' to the end

        for logic_form in logic_forms:
            if logic_form.strip() != "":
                if args.debug_mode:
                    print()
                    # print("The diagram logic form is", logic_form)
                # try:
                    # parse_tree = parser.parse(logic_form) # ['Equals', ['LengthOf', ['Line', 'A', 'C']], '10']
                    # parser.dfsParseTree(parse_tree)
                # except Exception as e:
                    # if args.debug_mode:
                        # print("\033[0;0;41mError:\033[0m", repr(e))

    # Parse text logic forms
    target = None
    text_logic_forms = text_parser["text_logic_forms"]
    for text in text_logic_forms:
        # if args.debug_mode:
            # print()
            # print("The text logic form is", text)
        if text.find('Find') != -1:
            target = parser.findTarget(parser.parse(text)) # ['Value', 'A', 'C']
        else:
            res = parser.parse(text)
            parser.dfsParseTree(res)

    # if args.debug_mode:
        # print()
        # print("The predicting sequence is", order_lst)

    solver = LogicSolver_v2(parser.logic)
    solver.initSearch()
    answer, steps, step_lst, usage = solver.Search(target=target,
                                                   order_list=order_lst,
                                                   round_or_step=args.enable_round,
                                                   upper_bound=args.round_limit if args.enable_round else
                                                   args.step_limit,
                                                   enable_low_first=args.low_first)

    return answer, steps, step_lst, usage


def get_step_lst(num, list1):
    para_lst = get_forms()
    index = num
    args = para_lst[index]['args']
    text_logic_form = para_lst[index]['text_logic_forms']
    diagram_logic_form = para_lst[index]['diagram_logic_forms']
    answer, steps, step_lst, usage = None, 0, [], 0

    args.debug_mode = True
    if args.debug_mode:
        answer, steps, step_lst, usage = solve_problem(args, text_logic_form, diagram_logic_form, list1)
    else:
        try:
            answer, steps, step_lst, usage = func_timeout(args.time_limit, solve_problem,
                                                           kwargs=dict(args=args,
                                                                 text_parser=text_logic_form,
                                                                 diagram_parser=diagram_logic_form,
                                                                 order_lst=list1))
        except FunctionTimedOut:
            pass
        except Exception as e:
            if args.debug_mode:
                print("\033[0;0;41mError:\033[0m", repr(e))

    print("answer:", answer)
    # print(answer, step_lst, usage)
    # answer, steps, step_lst, usage = solve_problem(args, text_logic_form, diagram_logic_form, list1)

    return answer, step_lst, usage


def single_point_crossover(lista, listb):
    if len(lista) == 0 or len(listb) == 0:
        print("error:父代出现空")
    sizes = min(len(lista), len(listb))
    if sizes == 1:
        cp = 1
    else:
        cp = np.random.randint(1, sizes)
    listc = lista[cp+1:len(lista)]
    lista = lista[1:cp] + listb[cp+1:len(listb)]
    listb = listb[1:cp] + listc
    if len(lista) == 0:
        print("error:子代为空")
        return listb
    else:
        return lista
"""
    if get_fitness(problem_number, lista) > get_fitness(problem_number, listb):
        return lista
    else:
        return listb
"""


def multi_point_crossover(lista, listb):
    p1 = len(lista)
    p2 = len(listb)
    sizes = min(p1, p2)
    if sizes == 1:
        cp1 = 1
        cp2 = 1
    elif sizes == 0:
        print('error:父代为空')
    else:
        cp1 = np.random.randint(1, sizes)
        cp2 = np.random.randint(1, sizes)
    if cp2 > cp1:
        cp2 += 1
    else:
        temp = cp1
        cp1 = cp2
        cp2 = temp
    listc = lista[cp1+1: cp2]
    lista = lista[1: cp1] + listb[cp1+1: cp2] + lista[cp2+1: p1]
    listb = listb[1: cp1] + listc + listb[cp2+1: p2]
    if len(lista) == 0:
        print('error:子代为空')
        return listb
    else:
        return lista
"""
    if get_fitness(problem_number, lista) > get_fitness(problem_number, listb):
        return lista
    else:
        return listb
"""


def uniform_crossover(lista, listb, p):
    sizes = min(len(lista), len(listb))
    if sizes == 0:
        print('error:父代为空')
    if sizes == 1:
        return lista
    for index in range(sizes):
        x1 = np.random.rand()
        if x1 <= p:
            temp = lista[index]
            lista[index] = listb[index]
            listb[index] = temp
    if len(lista) == 0:
        print('error:子代为空')
        return listb
    else:
        return lista
"""
    if get_fitness(problem_number, lista) > get_fitness(problem_number, listb):
        return lista
    else:
        return listb
"""


def standard_mutation(lista):
    sizes = len(lista)
    if sizes == 0:
        print('error:父代为空')
    if sizes == 1:
        return lista
    ind = np.random.randint(1, sizes)
    number = lista[ind]
    variation_number = counterfactual_variation(number)
    # pope = np.random.randint(1, 18)
    lista[ind] = variation_number
    if len(lista) == 0:
        print('error:子代为空')
    return lista


def modified_pairwise_interchange_mutation(lista):
    sizes = len(lista)
    if sizes == 0:
        print('error:父代为空')
    if sizes == 1:
        return lista
    ind1 = np.random.randint(1, sizes)
    ind2 = np.random.randint(1, sizes)
    number1 = lista[ind1]
    number2 = lista[ind2]
    pop1 = counterfactual_variation(number1)
    # pop1 = np.random.randint(1, 18)
    pop2 = counterfactual_variation(number2)
    # pop2 = np.random.randint(1, 18)
    lista[ind1] = pop1
    lista[ind2] = pop2
    if len(lista) == 0:
        print('error:子代为空')
    return lista


def get_fitness(num, lists):

    answer, step_list, usage = get_step_lst(num, lists)
    print("step list:", step_list)
    print("usage:", usage)

    if answer is not None:
        fitness = 1
    else:
        k_punish = 0.2
        # print("len:", len(lists))
        theorem_usage = usage / len(step_list)
        # fitness = theorem_usage

        most_freq = max(step_list, key=step_list.count)
        times = step_list.count(most_freq)
        punish = k_punish * times / len(step_list)
        fitness = theorem_usage - punish

    return fitness


if __name__ == "__main__":
    sin_CROSSOVER_RATE = 0.2
    mul_CROSSOVER_RATE = 0.3
    uni_CROSSOVER_RATE = 0.3
    st_MUTE_RATE = 0.1
    MPI_MUTE_RATE = 0.1
    GENERATION = 100
    standard_fitness = 0.3
    start = time.time()
    # final_seq = {}
    # 读入数据
    SEQ_PATH_1 = '../at_seq_pretreat_1.json'
    SEQ_PATH_2 = '../at_seq_pretreat_2.json'
    with open(os.path.join(SEQ_PATH_1), 'r') as f:
        data_1 = json.load(f)
    with open(os.path.join(SEQ_PATH_2), 'r') as f:
        data_2 = json.load(f)

    for problem_number in range(2401, 2401):
        final_path = '../GA_seq2/' + str(problem_number) + '.json'
        final_seq = {'id': problem_number, "seq": []}
        final_sequence = list()
        final_fitness = 0
        parent = list()
        parent.append(data_1[str(problem_number)]['seq'])
        parent.append(data_2[str(problem_number)]['seq'])

        for i in range(GENERATION):
            size = len(parent)
            loc1 = np.random.randint(0, size)
            loc2 = np.random.randint(0, size)
            listA = parent[loc1]
            listB = parent[loc2]
            if len(listA) == 0 or len(listB) == 0: continue

            x = np.random.rand()
            if x < 0.2:
                operation = single_point_crossover(listA, listB)
            elif x < 0.5:
                operation = multi_point_crossover(listA, listB)
            elif x < 0.8:
                operation = uniform_crossover(listA, listB, uni_CROSSOVER_RATE)
            elif x < 0.9:
                operation = standard_mutation(listA)
            else:
                operation = modified_pairwise_interchange_mutation(listA)
            print(operation)

            child_fitness = get_fitness(problem_number, operation)

            if child_fitness == 1:
                final_seq['seq'] = operation
                break
            elif child_fitness > standard_fitness:
                parent.append(operation)
                if child_fitness > final_fitness:
                    final_fitness = child_fitness
                    print(operation)
                    final_sequence = operation
                    print("final_seq:", final_sequence)
                    final_seq['seq'] = final_sequence
            print("generation:", i, "fitness:", child_fitness, "pop:", len(parent))
        print("num:", problem_number)
        final_seq['id'] = problem_number
        print(final_seq)
        print(parent)
        with open(final_path, 'w') as f:
            json.dump(final_seq, f, indent=2, separators=(',', ': '))
    end = time.time()
    print("running time；", end-start)
