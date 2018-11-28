import matplotlib as plt
import os.path as op
import operator

from independent_functions import combine_score, get_subdirs, printbar,get_res_subdirs


def list_same(_list):
    """checks if contains only the same values

    :param _list: a list
    :return: True if it does, else False
             False for empty list
    """
    if _list == []:
        return False
    for inst in _list:
        if inst != _list[0]:
            return False
    return True

def list_len_same(_list1, _list2):
    return len(_list1) == len(_list2)


class DataAccumulator:
    def __init__(self):
        self.Datasets = []
        self.State = 'enter'
        self.algtypes = ['ppa', 'hc', 'sa', 'rc']
        self.action_options = ['enter data', 'select_single_result', 'select_multiple_results', 'select_algorithm_comparison', 'plot_single_result', 'plot_multiple_results', 'plot_algorithm_comparison', 'plot', 'quit']
        self.one = None
        self.multiple = None
        self.compare1 = None
        self.compare2 = None

    def __str__(self):
        base = ''
        dataset_header = "Entered datasets:\n"
        dataset_str = "\n".join([ds.algtype + '\t' + ds.fname for ds in self.Datasets]) + '\n'
        base += dataset_header + dataset_str
        if self.one:
            single_str = self.one.algtype + '\t' + self.one.fname + '\n'
            single_header = "Current single Data:\n"
            base += single_header + single_str
        if self.multiple:
            multiple_header = "datasets selected for aggregate plot:\n"
            multiple_str = "\n".join([ds.algtype + '\t' + ds.fname for ds in self.multiple]) + '\n'
            base += multiple_header + multiple_str
        if self.compare1:
            comparison_header = "datasets selected for aggregate comparison:\n"
            comp1_header = '1st selection:\n'
            comp1_str = "\n".join([ds.algtype + '\t' + ds.fname for ds in self.compare1]) + '\n'
            comp2_header = '2nd selection:\n'
            comp2_str = "\n".join([ds.algtype + '\t' + ds.fname for ds in self.compare2]) + '\n'
            base += comparison_header + comp1_header, comp1_str + comp2_headerc + comp2_str
        return base

    def auto_select(self, cd):
        for inst in get_res_subdirs(cd):
            while inst[-4:] == '.tsv':
                try:
                    algtype = self.algtypes[int(input_select(self.algtypes,"which algorithms was used?"))]
                    self.add_data(algtype, op.join(cd,inst))
                    return True
                except IndexError:
                    print('that in not an eligible algorithm')
        return False

    def select_compare(self):
        names = [inst.fname for inst in self.Datasets]
        choices1 = []
        choices2 = []
        while True:
            num_choices1 = input_select(self.Datasets, "which datasets are the first part of your comparison")
            num_choices1 = [int(inst) for inst in num_choices1.split(',')]
            choices1 = [self.Datasets[i] for i in num_choices1]
            if list_same([inst.algtype for inst in choices1]):
                break
            else:
                print('not all of your selections come from the same' \
                      'algorithm, if you want to compare multiple algorithms'\
                      'select these at the next part')
        while True:
            num_choices2 = input_select(self.Datasets, "which datasets are the first part of your comparison")
            num_choices2 = [int(inst) for inst in num_choices2.split(',')]
            choices2 = [self.Datasets[i] for i in num_choices2]
            if list_same([inst.algtype for inst in choices2]):
                break
            else:
                print('not all of your selections come from the same' \
                      'algorithm, please retry')
        if list_len_same(choices1, choices2):
            return choices1, choices2
        else:
            question_datalength = input('your comparison datasets are of differing length, do you want to re-enter? (y/n)')
            if question_datalength == 'y':
                return self.select_compare()
            elif question_datalength == 'n':
                return choices1, choices2


    def select_multiple(self):
        choices = []
        while True:
            num_choices = input_select(self.Datasets, "which datasets are the first part of your comparison")
            num_choices = [int(inst) for inst in num_choices.split(',')]
            choices = [self.Datasets[i] for i in num_choices]
            if list_same([inst.algtype for inst in choices]):
                break
            else:
                print('not all of your selections come from the same' \
                      'algorithm, if you want to compare multiple algorithms'\
                      'select these at the next part')
        return choices


    def select_one(self):
        choice = input_select(self.Datasets, "which dataset do you want to plot")
        return self.Datasets[choice]


    def add_data(self, _type, file):
        if _type == 'ppa':
            self.Datasets.append(PPA_Data(file))
        elif _type == 'hc':
            self.Datasets.append(HC_Data(file))
        elif _type == 'rc':
            self.Datasets.append(RC_Data(file))
        elif _type == 'sa':
            raise NotImplementedError("simulated annealing data hangling has not yet been Implemented")


    def action(self, act_type):
        if act_type == 'enter data':
            print_dirwarning = False
            choice = ''
            while choice != 'a':
                basedir = op.curdir
                cd = op.join(op.curdir, 'circuit_map_git')
                while True:
                    if not op.isdir(cd):
                        cd = op.dirname(cd)
                        print_dirwarning = True
                    sdrs = get_res_subdirs(cd)
                    if self.auto_select(cd):
                        cd = op.dirname(cd)
                        continue
                    choice = input_select(sdrs, "select a subdirectory of file\n", extra=['all_entered', 'quit', 'up'], dirwarning=print_dirwarning)

                    print(choice)
                    if choice == 'a':
                        break
                    elif choice == 'u':
                        cd = op.dirname(cd)
                        continue
                    elif choice == 'q':
                        if input('are you sure? (y/n)\nThis abandons the plotting') == 'y':
                            exit(0)
                        else:
                            continue
                    elif choice == 'qq':
                        exit(0)
                    cd = op.join(cd, sdrs[int(choice)])
                    if cd[-4:] == '.tsv':
                        algtype = self.algtypes[int(input_select(self.algtypes,
                                               "which algorithms was used?"))]
                        self.add_data(algtype, cd)
                        cd = op.dirname(cd)
        elif act_type == 'select_single_result':
            self.one = self.select_one()
        elif act_type == 'select_multiple_results':
            self.multiple = self.select_multiple()
        elif act_type == 'select_algorithm_comparison':
            self.compare1, self.compare2 = self.select_compare()
        elif act_type == 'plot_single_result':
            self.plot_one()
        elif act_type == 'plot_multiple_results':
            self.plot_multiple()
        elif act_type == 'plot_algorithm_comparison':
            self.plot_compare()
        elif act_type == 'quit':
            if input('are you sure you want to quit? (y/n)') == 'y':
                exit(0)


    def mainloop(self):
        print(str(self))
        self.action('enter data')
        printbar(30)
        while True:
            print(str(self))
            ia = input_select(self.action_options, 'what would you like to do?', extra=['quit'])
            if ia == 'qq':   #quick quit
                exit(0)
            if ia == 'c':    #clear
                self.Datasets = []
            self.action(self.action_options[int(ia)])


class HC_Data:
    def __init__(self, file):
        aps = self.hc_gather(file)
        self.cs = aps[0]
        self.ls = aps[1]
        self.swaps = aps[2]
        self.its = aps[3]
        self.algtype = 'hc'
        self.fname = file

    def hc_gather(self,file):
        cons = []
        lens = []
        swaps = []
        with open(file) as f:
            for line in f:
                sl = line.split(',')
                sl1 = sl[1].split('\t')
                cons.append(int(sl[0]))
                lens.append(int(sl1[0]))
                if len(sl1) > 1:
                    swaps.append(int(sl1[1]))
        if len(swaps) > 0:
            return cons, lens, swaps, len(cons)
        else:
            return cons, lens, [1 for i in range(len(cons))], len(cons)

class RC_Data:
    def __init__(self, file):
        aps = self.rc_gather(file)
        self.cs = aps[0]
        self.ls = aps[1]
        self.its = aps[2]
        self.algtype = 'rc'
        self.fname = file

    def rc_gather(self,file):
        cons = []
        lens = []
        swaps = []
        with open(file) as f:
            for line in f:
                sl = line.split(',')
                cons.append(int(sl[0]))
                lens.append(int(sl[1]))
        return cons, lens, len(cons)

    def __add__(self, other):



class PPA_Data:
    def __init__(self, file):
        aps = self.ppa_gather(file)   #aspects
        #c_ms, c_bs, l_ms, l_bs, fe_cs, fe_ls, fef, gens, fes
        self.gc_ms = aps[0]  # generation_connection_means
        self.gc_bs = aps[1]  # generation_connection_maxima
        self.gl_ms = aps[2]  # generation_length_means
        self.gl_bs = aps[3]  # generation_best_lengths s.n.: length of best
                             # solution (not shortest length overall)
        self.fe_cs = aps[4]  # function evaluation connection score
        self.fe_ls = aps[5]  # function evaluation length score
        self.fes = aps[6]    # number of function evaluations
        self.gens = aps[7]   # numbe rof generations
        self.fetss = aps[8]   # fuction evaluations total scores
        self.algtype = "ppa"
        self.fname = file


    def ppa_gather(self, file):
        gc_ms = []  # generation_connection_means
        gc_bs = []  # generation_connection_maxima
        gl_ms = []  # generation_length_means
        gl_bs = []  # generation_best_lengths s.n.:

        fe_cs = []  # function evaluation connection score
        fe_ls = []  # function evaluation length score

        with open(file) as f:
            gt_cs = []  # temp_connections
            gt_ls = []  # temp_lengths
            for line in f:
                if line[0] == "#":
                    gc_bs.append(max(gt_cs))
                    gc_ms.append(mean(gt_cs))

                    cl = [combine_score(gt_cs[i], gt_ls[i])
                          for i in range(len(gt_cs))]
                    ind, max_value = max(enumerate(cl),
                                               key=operator.itemgetter(1))
                    print(ind, max_value)
                    gl_bs.append(gt_ls[ind])
                    gl_ms.append(mean(gt_ls))
                    gt_cs = []
                    gt_ls = []
                else:
                    sl = line.split(',')
                    gt_cs.append(int(sl[0]))
                    gt_ls.append(int(sl[1]))
                    fe_cs.append(int(sl[0]))
                    fe_ls.append(int(sl[1]))

        gens = len(gc_ms)  # number of generations
        fes = len(fe_cs)   # number of function evaluations
        # fets = function evaluation total scores
        fets = [combine_score(fe_cs[i], fe_ls[i]) for i in
                          range(len(fe_cs))]

        return gc_ms, gc_bs, gl_ms, gl_bs, fe_cs, fe_ls, fes, gens, fets


def input_select(_list, askstr, extra=None, dirwarning=False):
    for i in range(len(_list)):
        print(i, _list[i])
    if extra:
        for i, item in enumerate(extra):
            print(extra[i][0], item)
    if dirwarning:
        print('your last choice was not a directory and no eligible file\n Try again')
    return input(askstr)

def mean(_list):
    return sum(_list)/len(_list)




if True:
    da = DataAccumulator()
    da.mainloop()




