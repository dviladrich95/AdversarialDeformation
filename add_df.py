import pandas as pd
import os
import numpy as np
import math
import ast

sigma_list = [ math.pow(2,i) for i in range(8)]

for sigma in sigma_list:

    test_case = 'mnist'

    data_dict={}
    data_dict_sum={}

    # for key in def_data.keys():
    #     data_dict[key] = def_data[key].tolist()


    file_name=os.path.join('saved_results_1000',test_case+str(sigma).zfill(3))
    file_name_sum=file_name+ '_sum'

    df = pd.read_csv(file_name,sep='\t')
    df_sum = pd.read_csv(file_name_sum,sep='\t')


    a0 = df['1'][0].strip('][').split(', ')
    a1 = df['1'][1].strip('][').split(', ')
    a2 = df['1'][2].strip('][').split(', ')
    a3 = df['1'][3].strip('][').split(', ')
    a4 = df['1'][4].strip('][').split(', ')
    a5 = df['1'][5].strip('][').split(', ')


    data_dict['deformed_labels'] = np.asarray([ int(i) for i in a0])
    data_dict['original_labels'] = np.asarray([ int(i) for i in a1])
    data_dict['norms'] = np.asarray([ float(i) for i in a2])
    data_dict['iterations'] = np.asarray([ int(i) for i in a3])
    data_dict['overshot'] = np.asarray([ bool(i) for i in a4])
    data_dict['same_label'] = np.asarray([ bool(i) for i in a5])

    data_dict_sum['test_case'] = test_case
    data_dict_sum['sigma'] = sigma
    data_dict_sum['def_suc_rate'] = np.sum(data_dict['same_label'])/data_dict['same_label'].shape[0]
    data_dict_sum['avg_iter'] = np.sum(data_dict['iterations'])/data_dict['iterations'].shape[0]
    data_dict_sum['norm'] = np.sum(data_dict['norms'])/data_dict['norms'].shape[0]


    df = pd.DataFrame.from_dict(data_dict)
    df_sum = pd.DataFrame.from_dict(data_dict_sum)

    df.to_csv(file_name, sep='\t')
    df_sum.to_csv(file_name_sum, sep='\t')
