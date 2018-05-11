import csv, matplotlib,os, copy
import matplotlib.pyplot as plt

class algo_plots:

    def __init__(self):
        self.hc_dir = 'HC/'
        self.hc_dct = self.get_data()
        self.hc_plot()

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
                print(file)
                for idx,row in enumerate(csv_rd):
                    cur_con, cur_len = int(row[0]), int(row[1][0:4])
                    if cur_con > high_connect or (cur_con == high_connect and cur_len < high_len):
                        high_connect, high_len = copy.copy(cur_con), copy.copy(cur_len)
                        h_val = row[0]+'.'+row[1][0:4]
                        it_lst.append(idx)
                        score_lst.append(float(h_val))
                result_dict[i]['iter'], result_dict[i]['score'] = it_lst, score_lst
        print(result_dict)
        return result_dict

    def hc_plot(self):

        for key,item in self.hc_dct.items():
            print('jeej')
            plt.plot(item['iter'],item['score'])
            plt.ylabel('score (connections.length)')
            plt.xlabel('iterations')
        plt.show()



algo_plots()
