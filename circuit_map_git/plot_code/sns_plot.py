import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def get_data(file_name):

    with open(file_name) as csvfile:
        csv_rd = csv.reader(csvfile)
        gen_count = 1
        gen_dict = {1:[]}
        for row in csv_rd:
            print(row)
            if row[0].startswith('#'):
                gen_count += 1
                gen_dict[gen_count] = []
            else:
                score = int(row[0])+(1-(int(row[1])/10000))
                gen_dict[gen_count].append(score)

        # print(gen_dict)

    pd_dict = {'gen':[],'max':[],'mean':[],'min':[],'i':[]}

    i = 0
    for key, value in gen_dict.items():
        print(key, value)
        if value:
            val_len = len(value)
            mean = sum(value)/val_len
            max_val,min_val = max(value), min(value)
            print(mean, max_val)
            pd_dict['gen'].append(key)
            pd_dict['max'].append(max_val)
            pd_dict['mean'].append(mean)
            pd_dict['min'].append(min_val)
            pd_dict['i'].append(i)
            i += val_len

    df = pd.DataFrame(data=pd_dict)
    print(df)
    return df

def plottus(df, color):
    sns.tsplot(df['min'], df['i'],color=color)
    sns.tsplot(df['max'], df['i'],color=color)
    sns.tsplot(df['mean'], df['i'],color=color)
    plt.fill_between(df['i'], df['min'], df['max'], color=color, alpha='0.5')

def plottus_sd(df, color):
    # sns.tsplot(df['min'], df['i'],color=color)
    sns.tsplot((df['max']-df['mean']), df['i'],color=color)
    sns.tsplot((df['min']-df['mean']), df['i'],color=color)
    plt.fill_between(df['i'], (df['max']-df['mean']), (df['min']-df['mean']), color=color, alpha='0.5')

df1 = get_data('PPA/ppa_1.tsv')
df4 = get_data('PPA/ppa_4.tsv')

sns.set_style("darkgrid")
plottus(df1, 'red')
plottus(df4, 'blue')
plt.show()
