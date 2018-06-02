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

df = get_data('PPA/ppa.tsv')

sns.set_style("darkgrid")
ax1 = sns.tsplot(df['min'], df['i'])
ax2 = sns.tsplot(df['max'], df['i'])
ax3 = sns.tsplot(df['mean'], df['i'])
plt.fill_between(df['i'], df['min'], df['max'], color='grey', alpha='0.5')
plt.show()
