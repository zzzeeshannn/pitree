import numpy as np
import matplotlib.pyplot as plt
import os

def read_log(file_name):
    data = []
    with open(file_name) as f:
        while True:
            lines = f.readline()
            if not lines:
                break
            line = lines.split("\t")
            if len(line) > 1 :
                for i in range(len(line)-1):
                    line[i+1] = line[i+1].strip()
                line[-1].replace('\n','')
                for i in range(len(line)):
                    line[i] = float(line[i])
                data.append(line)
    data = np.array(data,dtype = float)
    return data

def read_folder(dirname):
    logdata = []
    #dirname = './graph/pensieve_fcc_dt_fcc_100n'
    files = os.listdir(dirname)
    files.sort()
    for file in files:
        file_name = dirname+'/'+file
        logdata.append(read_log(file_name))
    logdata = np.array(logdata)
    return logdata

def trace_compare(traces):
    traces_bdw_avg = []
    traces_bdw_std = []
    for trace in traces:
        allavg_bdw = []
        allstd_bdw = []
        for i in range(len(trace)):# trace files number
            avg_bdw = 0.
            std_bdw = 0.
            for j in range(len(trace[i])): # each row
                avg_bdw += trace[i][j][1]
            avg_bdw = avg_bdw/len(trace[i])
            for j in range(len(trace[i])):
                std_bdw += np.square(trace[i][j][1]-avg_bdw)
            std_bdw = np.sqrt((std_bdw/len(trace[i])))
            allavg_bdw.append(avg_bdw)
            allstd_bdw.append(std_bdw)
        traces_bdw_avg.append(np.array(allavg_bdw))
        traces_bdw_std.append(np.array(allstd_bdw))
    traces_bdw_avg = np.array(traces_bdw_avg)
    traces_bdw_std = np.array(traces_bdw_std)
    return traces_bdw_avg,traces_bdw_std

def log_data(colnum,logfiles):
    all_avg = []
    for i in range(len(logfiles)):
        avg = 0.
        for j in range(len(logfiles[i])):
            avg += logfiles[i][j][colnum]
        avg = avg/len(logfiles[i])
        all_avg.append(avg)
    all_avg = np.array(all_avg)
    return all_avg

def log_compare_graph(logfolders,colnum):
    metric_records = []
    for folder in logfolders:
        metric_records.append(log_data(colnum,folder))
    return metric_records


if __name__ == '__main__':
#########################################################################
# For different abr algorithms
    logfolders = []
    #dirname0 = './graph/pensieve_ori_oboe'
    dirname1 = './graph/pensieve_ori_oboe'
    dirname2 = './graph/robustmpc_ori_oboe'
    dirname3 = './graph/hotdash_ori_oboe'
    #logfolders.append(read_folder(dirname0))
    logfolders.append(read_folder(dirname1))
    logfolders.append(read_folder(dirname2))
    logfolders.append(read_folder(dirname3))
    graph_data = log_compare_graph(logfolders, 6)
    plt.figure()
    plt.title('3 original abr testing on QoE')
    hist = []
    binedge = []
    cdf = []
    for i in range(len(logfolders)):
        hist.append(np.histogram(graph_data[i])[0])
        binedge.append(np.histogram(graph_data[i])[1])
        cdf.append(np.cumsum(hist[i] / sum(hist[i])))
        plt.plot(binedge[i][1:], cdf[i], '-*')
    plt.legend(['pensieve','robustmpc','hotdash'])
    plt.savefig('./graph/abr_algorithms_comparison_ori')
    plt.show()
###############################################################

#########################################################################
#For different decision tree
    logfolders = []
    # ..................
    dirname0 = './graph/pensieve_ori_oboe'
    dirname1 = './graph/pensieve_fcc_dt_oboe_200n'
    dirname2 = './graph/pensieve_fcc_ada_oboe_200n'
    dirname3 = './graph/pensieve_fcc_xgb_oboe_200n'
    #dirname4 = './graph/pensieve_fcc_id3_oboe_200n'
    logfolders.append(read_folder(dirname0))
    logfolders.append(read_folder(dirname1))
    logfolders.append(read_folder(dirname2))
    logfolders.append(read_folder(dirname3))
    #logfolders.append(read_folder(dirname4))
    graph_data = log_compare_graph(logfolders, 6)
    plt.figure()
    plt.title('Pensieve based decision tree testing on QoE')
    hist = []
    binedge = []
    cdf = []
    for i in range(len(logfolders)):
        hist.append(np.histogram(graph_data[i])[0])
        binedge.append(np.histogram(graph_data[i])[1])
        cdf.append(np.cumsum(hist[i] / sum(hist[i])))
        plt.plot(binedge[i][1:], cdf[i], '-*')
    plt.legend(['original','cart', 'adaboost', 'xgboost'])
    plt.savefig('./graph/decision_tree_test_rebuffertime_oboe')#................
    plt.show()
###############################################################

##########################################
# For traces on dt
    logfolders = []
    dirname1 = './graph/pensieve_fcc_dt_fcc_200n'
    dirname2 = './graph/pensieve_fcc_dt_norway_200n'
    dirname3 = './graph/pensieve_fcc_dt_oboe_200n'
    logfolders.append(read_folder(dirname1))
    logfolders.append(read_folder(dirname2))
    logfolders.append(read_folder(dirname3))
    graph_data = log_compare_graph(logfolders,6)
    plt.figure()
    plt.title('Pensieve_fcc testing on different traces')
    hist = []
    binedge = []
    cdf = []
    for i in range(len(logfolders)):
        hist.append(np.histogram(graph_data[i])[0])
        binedge.append(np.histogram(graph_data[i])[1])
        cdf.append(np.cumsum(hist[i]/sum(hist[i])))
        plt.plot(binedge[i][1:],cdf[i],'-*')
    plt.legend(['fcc', 'norway', 'oboe'])
    plt.savefig('./graph/pensieve_fcc_traces_test')
    plt.show()

# For traces on ori
    logfolders = []
    dirname1 = './graph/pensieve_ori_fcc'
    dirname2 = './graph/pensieve_ori_norway'
    dirname3 = './graph/pensieve_ori_oboe'
    logfolders.append(read_folder(dirname1))
    logfolders.append(read_folder(dirname2))
    logfolders.append(read_folder(dirname3))
    graph_data = log_compare_graph(logfolders,6)
    plt.figure()
    plt.title('Pensieve_ori testing on different traces')
    hist = []
    binedge = []
    cdf = []
    for i in range(len(logfolders)):
        hist.append(np.histogram(graph_data[i])[0])
        binedge.append(np.histogram(graph_data[i])[1])
        cdf.append(np.cumsum(hist[i]/sum(hist[i])))
        plt.plot(binedge[i][1:],cdf[i],'-*')
    plt.legend(['fcc', 'norway', 'oboe'])
    plt.savefig('./graph/pensieve_ori_traces_test')
    plt.show()
##############################################

#############################################
    # # For traces distribution
    traces = []
    dir_trace_fcc = './sim_fcc'
    fcc_traces = read_folder(dir_trace_fcc)
    traces.append(fcc_traces)
    dir_trace_norway = './sim_norway'
    norway_traces = read_folder(dir_trace_norway)
    traces.append(norway_traces)
    dir_trace_oboe ='./sim_oboe'
    oboe_traces = read_folder(dir_trace_oboe)
    traces.append(oboe_traces)
    traces_bdw_avg,traces_bdw_std = trace_compare(traces)
    # Make graph
    # avg
    plt.figure()
    plt.title('CDF for average bandwidth')
    hist0,binedge0 = np.histogram(traces_bdw_avg[0])
    cdf0 = np.cumsum(hist0/sum(hist0))
    plt.plot(binedge0[1:],cdf0,'-*')
    hist1,binedge1 = np.histogram(traces_bdw_avg[1])
    cdf1 = np.cumsum(hist1/sum(hist1))
    plt.plot(binedge1[1:],cdf1,'-*')
    hist2,binedge2= np.histogram(traces_bdw_avg[2])
    cdf2 = np.cumsum(hist2/sum(hist2))
    plt.plot(binedge2[1:],cdf2,'-*')
    plt.legend(['fcc','norway','oboe'])
    plt.savefig('./graph/trace_bw_avg')
    plt.show()

    # std
    plt.figure()
    plt.title('CDF for standard deviation bandwidth')
    hist0,binedge0 = np.histogram(traces_bdw_std[0])
    cdf0 = np.cumsum(hist0/sum(hist0))
    plt.plot(binedge0[1:],cdf0,'-*')
    hist1,binedge1 = np.histogram(traces_bdw_std[1])
    cdf1 = np.cumsum(hist1/sum(hist1))
    plt.plot(binedge1[1:],cdf1,'-*')
    hist2,binedge2= np.histogram(traces_bdw_std[2])
    cdf2 = np.cumsum(hist2/sum(hist2))
    plt.plot(binedge2[1:],cdf2,'-*')
    plt.legend(['fcc','norway','oboe'])
    plt.savefig('./graph/trace_bw_std')
    plt.show()
########################################









