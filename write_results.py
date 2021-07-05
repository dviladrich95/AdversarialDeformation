import pandas as pd
import os
import numpy as np
from deformation import Tnorm, gaussian_filter
import torch

def write_results(adversarial,test_case,norm_type,def_data,batch_size,sigma,l,mu,e_total):



    data_np=np.empty((7,batch_size))
    data_np_sum=np.empty(9)

    # for key in def_data.keys():
    #     data_dict[key] = def_data[key].tolist()


    file_name=os.path.join('results',test_case+'_'+norm_type+'_'+adversarial+'_s'+str(int(sigma)).zfill(3)+'_l'+str(int(l)).zfill(3)+'_mu'+str(int(mu)).zfill(3))

    file_name_sum=file_name+ '_sum'

    #0'deformed_labels'
    #1'original_labels'
    #2'norms'
    #3'iterations'
    #4'overshot'
    #5'same_label'

    data_np[0] = def_data['deformed_labels'].cpu().numpy()
    data_np[1] = def_data['original_labels'].cpu().numpy()
    data_np[2] = def_data['norms'].cpu().numpy()
    data_np[3] = def_data['iterations'].cpu().numpy()
    data_np[4] = data_np[0] != data_np[1]
    data_np[5] = e_total.cpu().numpy()
    vf = def_data['vector_fields']

    data_np_sum[0] = batch_size
    data_np_sum[1] = sigma
    data_np_sum[2] = np.sum(data_np[2])/data_np[2].shape[0] #mean norm
    data_np_sum[3] = np.sum(data_np[3])/data_np[3].shape[0] #mean iterations
    data_np_sum[4] = np.sum(data_np[4])/data_np[4].shape[0]*100 #adef success rate
    data_np_sum[5] = l
    data_np_sum[6] = mu
    data_np_sum[7] = np.sum(data_np[5])/data_np[5][data_np[5] != 0].shape[0]
    #smoother = gaussian_filter(sigma=16)
    #smooth_vf = smoother(torch.tensor(vf))
    #data_np_sum[8] = Tnorm(smooth_vf,'norm_type',l,mu)-data_np[2]

    df = pd.DataFrame(data_np)
    df_sum = pd.DataFrame(data_np_sum)

    df.to_csv(file_name, sep='\t')
    df_sum.to_csv(file_name_sum, sep='\t')
    return df