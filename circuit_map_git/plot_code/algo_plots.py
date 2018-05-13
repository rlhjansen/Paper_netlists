import csv, matplotlib,os, copy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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
        self.hc_plot()
        self.ppa_plot()
        self.rd_plot()

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
        return result_dict

    def hc_plot(self):

        for key,item in self.hc_dct.items():
            print(key)
            plt.plot(item['iter'],item['score'],label='hc{}'.format(key))
            plt.ylabel('score (connections.length)')
            plt.xlabel('iterations')
        for key,item in self.ppa_dct.items():
            plt.plot(item['sort'],'b-', linewidth=0.5,label='ppa{}'.format(key))
            plt.ylabel('score (connections.length)')
            plt.xlabel('iterations')
        plt.legend()
        plt.savefig('hc_plot.png')
        plt.gcf()


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
                        print(sort_lst)
                        temp_lst = []
                    else:
                        #cur_score = float(row[0]+'.'+row[1])
                        cur_score = combine_score(int(row[0]), int(row[1]))
                        score_lst.append(cur_score)
                        gen_lst.append(gen_count)
                        temp_lst.append(cur_score)
                # print(sort_lst)
                ppa_dct[i]['score'] = score_lst
                ppa_dct[i]['gen'] = gen_lst
                ppa_dct[i]['sort'] = sort_lst
        return ppa_dct

    def ppa_plot(self):

        for key,item in self.ppa_dct.items():
            # print('jeej')
            plt.plot(item['score'],'b-',linewidth=0.1)
            plt.ylabel('score (connections.length)')
            plt.xlabel('iterations')
        plt.savefig('ppa_plot.png')
        plt.gcf()


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

    def rd_plot(self):

        for key,item in self.rd_dct.items():
            print('jeej')
            density = gaussian_kde(item['score'])
            xs = np.linspace(40, 90, 1000)
            plt.plot(xs, density(xs), label=item['netlist'])
            #plt.plot(item['score'],'b-',linewidth=0.1)
            plt.ylabel('density')
            plt.xlabel('score (connections.length)')
            plt.title('density of random configuration scores')
            plt.legend()
        plt.savefig('rd_plot.png')
        plt.gcf()


algo_plots()
