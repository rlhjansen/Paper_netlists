import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv, os, sys

def get_ppa():

    pd_dict = {'ppa':[],'gen':[],'max':[],'mean':[],'min':[],'max_pop':[],'mean_pop':[],'min_pop':[],'max_el':[],'mean_el':[],'min_el':[],'i':[]}

    ppa_dir = os.listdir('PPA_yv')

    for i,file in enumerate(ppa_dir):
        ppa_nm = file.split('_')[1]
        # print(ppa_nm)
        with open('PPA_yv/'+file) as csvfile:
            csv_rd = csv.reader(csvfile)
            gen_count = 0
            gen_dict = {0:[]}
            for row in csv_rd:
                # print(row)
                if row[0].startswith('#'):
                    gen_count += 1
                    gen_dict[gen_count] = []
                else:
                    score = int(row[0])+(1-(int(row[1])/10000))
                    gen_dict[gen_count].append(score)

            # print(gen_dict)


        i = 0
        select_val = []
        for key, value in gen_dict.items():
            # print(key, value)
            if value:
                val_len = len(value)
                mean = sum(value)/val_len
                max_val,min_val = max(value), min(value)
                select_val += value
                sort_val = sorted(select_val, reverse=True)
                pop_len = len(sort_val)
                pd_dict['ppa'].append(ppa_nm)
                pd_dict['gen'].append(key)
                pd_dict['max'].append(max_val)
                pd_dict['mean'].append(mean)
                pd_dict['min'].append(min_val)
                pd_dict['max_pop'].append(sort_val[0])
                pd_dict['mean_pop'].append(sum(sort_val)/pop_len)
                pd_dict['min_pop'].append(sort_val[-1])
                pd_dict['i'].append(i)
                select_val = sort_val[:30]
                pd_dict['max_el'].append(select_val[0])
                pd_dict['mean_el'].append(sum(select_val)/len(select_val))
                pd_dict['min_el'].append(select_val[-1])
                i += val_len
    df = pd.DataFrame(data=pd_dict)
    df.to_pickle('ppa_df.pickle')
    df.to_csv('ppa_df.csv')
    # print(df)
    return df

def get_hc():
    hc_dct = {'hc':[],'i':[],'score':[],'subj':[]}
    hc_dir = os.listdir('HC_yv')

    for i,file in enumerate(hc_dir):
        hc_nm = file.split('_')[1]
        # print(ppa_nm)
        with open('HC_yv/'+file) as csvfile:
            csv_rd = csv.reader(csvfile)
            for i, row in enumerate(csv_rd):
                score = int(row[0])+(1-(int(row[1][0:4])/10000))
                hc_dct['hc'].append(hc_nm)
                hc_dct['i'].append(i)
                hc_dct['score'].append(score)
                hc_dct['subj'].append(0)

    df = pd.DataFrame(data=hc_dct)
    df.to_pickle('hc_df.pickle')
    df.to_csv('hc_df.csv')
    return df




def best_med(df):
    ppa_lst = list(df.ppa.unique())
    score_lst = []
    for run in ppa_lst:
        jup = df.loc[df['ppa'] == run].sort_values('max_pop', ascending=True)
        score_lst.append(float(jup.max_pop.iat[-1]))
        # print(sorted(list(jup.max_pop.iat[-1])))
        # print(list(jup[['ppa','max_pop']].iloc[-1]))
        # print(jup)
    score_lst, ppa_lst = (list(t) for t in zip(*sorted(zip(score_lst, ppa_lst))))
    print(score_lst, ppa_lst)

def plottus(df_hc, df_ppa):

    # print(df_ppa)

    sns.set_style("darkgrid")
    df_ppa['subject'] = 0
    ppa_h = df_ppa.loc[df_ppa['ppa'] == '6']
    ppa_l = df_ppa.loc[df_ppa['ppa'] == '2']
    print(df_hc)
    hc_h = df_hc.loc[df_hc['hc'] == 5]
    hc_l = df_hc.loc[df_hc['hc'] == 1]
    print(hc_h)



    # print(ppa_h)
    # df.plot()
    # ax1 = sns.tsplot(time='i',condition='ppa', value='max_el',data=ppa_df,unit='subject')
    # plt.fill_between(df['i'], df['max_pop'], df['mean_pop'], color='red', alpha='0.5')
    # ax2 = sns.tsplot(time='i',condition='ppa', value='max_el',data=df,unit='subject')
    # fig, ax = plt.subplots()
    # labels = ['PPA', 'HC']
    ax = sns.tsplot(ppa_h['max_pop'], ppa_h['i'],color='blue')
    sns.tsplot(ppa_l['max_pop'], ppa_l['i'],color='blue')
    sns.tsplot(hc_h['score'], hc_h['i'],color='red')
    sns.tsplot(hc_l['score'], hc_l['i'],color='red')
    plt.fill_between(hc_l['i'], hc_h['score'], hc_l['score'], color='red', alpha='0.4')
    plt.fill_between(ppa_h['i'], ppa_h['max_pop'], ppa_l['max_pop'], color='blue', alpha='0.4')
    ax.set(xlabel='function evaluations', ylabel='objective function value')
    plt.legend(['PPA', 'HC'],loc='upper left', frameon=True)
    # plt.show()
    plt.savefig('FINAL_HC_PPA_PLOT.png')

    # sns.tsplot(df['mean_el'], df['i'],color='blue')
    # plt.fill_between(df['i'], df['max_el'], df['mean_el'], color='blue', alpha='0.5')

df_ppa = get_ppa()
# best_med(df_ppa)
df_hc = pd.read_pickle('hc_df_jeah.pickle')

# print(df_hc)
# print(df)
plottus(df_hc,df_ppa)
# best_med(df)


# plottus(df4, 'blue')
