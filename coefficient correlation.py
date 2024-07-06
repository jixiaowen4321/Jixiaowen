# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:01:55 2024

@author: Ji Xiaowen
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats  as stats


def read_log_df(file_):
    
    Df = pd.read_csv(file_, engine='c').drop(columns=['Unnamed: 0',
                                                      'sample_id'])
    Df.iloc[:, :-1] = np.log10(Df.iloc[:, :-1])
    Cols = list(Df.columns[:-1])
    
    return Df, Cols


def split_range(Cols):
    min_ = 100
    max_ = 500
    n = 9
    intervals = np.linspace(min_, max_, num=n)
    print(intervals)
    #print(intervals)
    L = [ [] for _ in range(n-1) ]
    L_cols = [ [] for _ in range(n-1) ]
    
    for i in range(n-1):
        int_ = pd.Interval(left=intervals[i],
                           right=intervals[i+1])
        l = []
        for col in Cols:
            col_ = float(col[:7])
            if col_ in int_:
                l.append(col)
            else:
                pass

        L[i] = l
        L_cols[i] = str(intervals[i]) + ' ~ ' + str(intervals[i+1])

    return L, L_cols



def corr_p(Df, Cols):
    corr = []
    p = []
    data = np.zeros((2, len(Cols)))
    
    for j in range(len(Cols)):
        col = Cols[j]
        coef, p = stats.pearsonr(Df[col],
                                 Df['ga_delivery'])
        data[0, j] = coef
        data[1, j] = p

    Corr_results = pd.DataFrame(data,
                                columns=Cols,
                                index=['Correlation Coefficients',
                                       'P-Values'])
    return Corr_results



def corr_mean(L, L_cols, Corr_results):

    Df_mean = pd.DataFrame()
    for k in range(len(L)):
        l = L[k]
        col_range = L_cols[k]
    
        Df_mean[col_range] = Corr_results[l].mean(axis=1)

    #print(df_mean)
    
    return Df_mean



def plot_(Index, L_cols, Df_mean):
    plt.rcParams.update({'font.size': 12})
    for index in Index:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.scatter(L_cols, Df_mean.loc[index, ], s=10)
        ax.plot(L_cols, Df_mean.loc[index, ])
        ax.set_ylabel(index, fontsize=15)
        plt.tight_layout()

        plt.savefig('./' + index + '.png')
    plt.show()


    return 0



def main():
    
    f = 'C:\Xiaowen\Data\data-processing\5. Preterm days and Mass relationship\1. New/CorPOS_GAdelivery_serum.csv'
    df, cols = read_log_df(f)
    #print(df)
    #print(cols)

    l, l_cols = split_range(cols)
    #print(l, l_cols)

    corr_results = corr_p(df, cols)

    df_mean = corr_mean(l, l_cols, corr_results)
    index = ['Correlation Coefficients', 'P-Values']

    plot_(index, l_cols, df_mean)

    return 0
    

    
if __name__ == "__main__":
    main()