import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def read_csv(csv_name):
    data = []
    with open(csv_name) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            for i in range(len(row)):
                row[i] = float(row[i])
            data.append(row)
    data = np.array(data)
    avg_data = np.mean(data,axis=0)
    return avg_data

def read_csv_p(csv_name):
    data = []
    with open(csv_name) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            for i in range(len(row)):
                row[i] = float(row[i])
            data.append(row)
    data = np.array(data)
    return data

if __name__ == '__main__':
    ###################################################
    # For different decision tree
    all_dt = []
    avg_data0 = read_csv('./time/pensieve_fcc_200.csv')
    avg_data1 = read_csv('./time/ada_pensieve_fcc_200.csv')
    avg_data2 = read_csv('./time/xgb_pensieve_fcc_200.csv')
    avg_data3 = read_csv('./time/id3_pensieve_fcc_200.csv')

    all_dt.append(avg_data0)
    all_dt.append(avg_data1)
    all_dt.append(avg_data2)
    all_dt.append(avg_data3)
    all_dt = np.array(all_dt)
    #
    labels = ['cart','adaboost','xgboost','id3']
    time1 = all_dt[:,0]
    time2 = all_dt[:,1]
    time3 = all_dt[:,2]

    x = np.arange(len(labels))
    width = 0.2
    fig,ax = plt.subplots()
    rects1 = ax.bar(x - width*2, time1, width, label='fitting time')
    rects2 = ax.bar(x - width+0.01, time2, width, label='visual play for teacher')
    rects3 = ax.bar(x + width+0.02, time3, width, label='time for teacher prediction')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig('./graph/decision_trees_training_time_per')
    plt.show()

    # For different abr
    all_abr = []
    avg_data0 = read_csv('./time/pensieve_fcc_100.csv')
    avg_data1 = read_csv('./time/robustmpc_fcc_100.csv')
    avg_data2 = read_csv('./time/hotdash_fcc_100.csv')
    all_abr.append(avg_data0)
    all_abr.append(avg_data1)
    all_abr.append(avg_data2)
    all_abr = np.array(all_abr)
    #
    labels = ['pensieve','robustmpc','hotdash']
    time1 = all_abr[:,0]
    time2 = all_abr[:,1]
    time3 = all_abr[:,2]

    x = np.arange(len(labels))
    width = 0.2
    fig,ax = plt.subplots()
    rects1 = ax.bar(x - width*2, time1, width, label='fitting time')
    rects2 = ax.bar(x - width+0.01, time2, width, label='visual play for teacher')
    rects3 = ax.bar(x +  width+0.02, time3, width, label='time for teacher prediction')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig('./graph/abr_algorithms_training_time_per')
    plt.show()
    ########################################################
    # For  pensieve precision
    precison1 = read_csv_p('./precision/pensieve_fcc_100.csv')
    precison2 = read_csv_p('./precision/pensieve_fcc_200.csv')
    precison3 = read_csv_p('./precision/pensieve_fcc_500.csv')
    precison4 = read_csv_p('./precision/pensieve_fcc_1000.csv')
    plt.figure()
    plt.title('Training process of prensieve')
    plt.plot(precison1)
    plt.plot(precison2)
    plt.plot(precison3)
    plt.plot(precison4)
    plt.legend(['N100','N200','N500','N1000'])
    plt.savefig('./graph/training_process_pensieve_fcc')
    plt.show()


