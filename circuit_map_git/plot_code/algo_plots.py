import csv, os, copy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statistics import median
import numpy as np

def combine_score(connections, length):
    frac_part = float(10000-length)/10000.
    return float(connections)+frac_part


class algo_plots:

    def __init__(self):
        self.hc_dir = 'HC/'
        self.ppa_dir = 'PPA/'
        self.rd_dir = 'RD/'
        self.hc_dct = self.hc_data()
        self.ppa_dct = self.ppa_data()
        self.rd_dct = self.rd_data()
        self.hc_plot(ppa_compare=True, ppa_median=True, ppa_best=True, all=False, best=True, worst=True, median=True)
        #self.ppa_plot()
        self.rd_plot(compare_ppa=False)
        self.get_best_ppa_iterdict()

    def hc_data(self):

        result_dict = {}
        hc_dir = os.listdir(self.hc_dir)

        for i,file in enumerate(hc_dir):
            result_dict[i] = {}
            # print('cur dir', file)
            it_lst,score_lst = [], []
            high_connect, high_len = 0, 0
            with open(self.hc_dir+file) as csvfile:
                csv_rd = csv.reader(csvfile)
                # print(file)
                h_val = 0.0000
                for idx,row in enumerate(csv_rd):
                    cur_con, cur_len = int(row[0]), int(row[1][0:4])
                    if cur_con > high_connect or (cur_con == high_connect and cur_len < high_len):
                        high_connect, high_len = copy.copy(cur_con), copy.copy(cur_len)
                        #h_val = row[0]+'.'+row[1][0:4]
                        h_val = combine_score(high_connect, high_len)
                    it_lst.append(idx)
                    score_lst.append(float(h_val))
                result_dict[i]['iter'], result_dict[i]['score'] = it_lst, score_lst
        # print(result_dict)
        best, bi = 0, None
        worst, wi = 100, None
        s_endpoints = sorted([result_dict[i]['score'][-1] for i in range(len(result_dict))])
        median, mi = s_endpoints[int(len(s_endpoints)/2)], None
        for i in range(len(result_dict)):
            if result_dict[i]['score'][-1] < worst:
                worst, wi = result_dict[i]['score'][-1], i
            if result_dict[i]['score'][-1] > best:
                best, bi = result_dict[i]['score'][-1], i
            if result_dict[i]['score'][-1] == median:
                mi = i
        self.besthc = bi
        self.worsthc = wi
        self.medianhc = mi
        return result_dict

    def get_best_ppa_iterdict(self, compare_median=True):
        for key, item in self.ppa_dct.items():
            print(key, max(item['score']))
        if compare_median:
            return [median([max(item['score']) for key, item in self.ppa_dct.items()])]
        else:
            return [max(item['score']) for key, item in self.ppa_dct.items()]



    def hc_plot(self, all=True, ppa_compare=False, ppa_median=False, ppa_best=True, best=False, worst=False, median=False):
        plt.clf()
        plt.figure(figsize=(10, 5))
        plt.ylabel('score')
        plt.xlabel('function evaluations')
        #plt.title('Algorithm comparison of hillclimber and plant propagation for netlist with 100 nets')
        if all:
            for key,item in self.hc_dct.items():
                #print(key)
                plt.plot(item['iter'],item['score'],label='hc{}'.format(key))
        if ppa_compare:
            if ppa_best:
                plt.plot(self.ppa_dct['best']['score'], 'r-', linewidth=0.5,label='best ppa', color="#feb24c")
            if ppa_median:
                plt.plot(self.ppa_dct['median']['score'], 'g-',linewidth=0.5,
                         label='median ppa', color="#9ebcda")
            if False:
                for key,item in self.ppa_dct.items():
                    plt.plot(item['sort'],'b-', linewidth=1.,label='sample ppa')
                    break
        if best:
            bi = self.besthc
            plt.plot(self.hc_dct[bi]['iter'], self.hc_dct[bi]['score'], linewidth=1.2, color="#de2d26", label='best hc')
        if worst:
            wi = self.worsthc
            plt.plot(self.hc_dct[wi]['iter'], self.hc_dct[wi]['score'], linewidth=1.2, label='worst hc')
        if median:
            mi = self.medianhc
            plt.plot(self.hc_dct[mi]['iter'], self.hc_dct[mi]['score'], linewidth=1.2, color="#756bb1",label='median hc')
        plt.legend()
        plt.savefig('hc_plot_compare.png')

    def ppa_data(self):

        ppa_dct = {}
        ppa_files = os.listdir(self.ppa_dir)
        for i, file in enumerate(ppa_files):
            ppa_name = (file.strip('.tsv').split('_')[-1])
            ppa_dct[i] = {}
            with open(self.ppa_dir+file) as csvfile:
                csv_rd = csv.reader(csvfile)
                score_lst, iter_lst, sort_lst = [], [], []
                gen_count, gen_lst = 1, []
                temp_lst = []
                for row in csv_rd:
                    if row[0].startswith('##'):
                        gen_count += 1
                        temp_lst.sort()
                        # print(temp_lst)
                        sort_lst += temp_lst
                        #print(sort_lst)
                        temp_lst = []
                    else:
                        #cur_score = float(row[0]+'.'+row[1])
                        cur_score = combine_score(int(row[0]), int(row[1]))
                        score_lst.append(cur_score)
                        gen_lst.append(gen_count)
                        temp_lst.append(cur_score)
                # print(sort_lst)
                ppa_dct[i]['score'] = score_lst
                ppa_dct[i]['max'] = max(score_lst)
                ppa_dct[i]['gen'] = gen_lst
                ppa_dct[i]['sort'] = sort_lst

        maxlist = [max(item['score']) for key, item in ppa_dct.items()]
        med = median(maxlist)
        mmax = max(maxlist)
        ind_med = maxlist.index(med)
        ind_max = maxlist.index(mmax)
        ppa_dct['median'] = ppa_dct[ind_med]
        ppa_dct['best'] = ppa_dct[ind_max]
        return ppa_dct

    def ppa_plot(self):
        plt.clf()
        for key,item in self.ppa_dct.items():
            # print('jeej')
            if key == 'best' or key == 'median':
                continue
            plt.plot(item['score'],'b-',linewidth=0.1)
            plt.ylabel('score (connections.length)')
            plt.xlabel('function evaluations')
        plt.savefig('ppa_plot.png')


    def rd_data(self):

        rd_dct = {}
        rd_files = os.listdir(self.rd_dir)
        netlists = ['netlist 1','netlist 3','netlist 4']
        for i, file in enumerate(rd_files):
            rd_name = (file.strip('.tsv').split('_')[-1])
            rd_dct[i] = {}
            with open(self.rd_dir+file) as csvfile:
                csv_rd = csv.reader(csvfile)
                score_lst, iter_lst = [],[]
                for row in csv_rd:
                    if row[0].startswith('##'):
                        continue
                    else:
                        #cur_score = float(row[0]+'.'+row[1])
                        cur_score = combine_score(int(row[0]), int(row[1]))
                        score_lst.append(cur_score)

                rd_dct[i]['score'] = score_lst
                rd_dct[i]['netlist'] = netlists[i]
        return rd_dct

    def rd_plot(self, compare_ppa=None, median_compare=True):
        plt.clf()
        #plt.figure(figsize=(10, 5))
        for key,item in self.rd_dct.items():
            print('jeej')
            density = gaussian_kde(item['score'])
            xs = np.linspace(40, 90, 10000)
            plt.plot(xs, density(xs), label='random samples')
            #plt.plot(item['score'],'b-',linewidth=0.1)
            break
        if compare_ppa:
            res_ppa = self.get_best_ppa_iterdict(compare_median=median_compare)
            for inst in res_ppa:
                plt.axvline(x=inst, color='g', linestyle=':', label='median ppa')

        plt.ylabel('density of 10.000 RS')
        plt.xlabel('score')
        #plt.title('score density in the search space, 10E4')
        plt.legend()
        plt.savefig('rd_plot.png')
        print('plotted random')


algo_plots()
