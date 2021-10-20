import os
os.environ['HOME'] ='/home/ohno'
import numpy as np
import pandas as pd
import time
import sys
from tqdm import tqdm
import argparse
import numpy as np
from scipy import signal
import scipy.spatial.distance as distance
import random
import math

INTERACTION_PATH = os.path.join(os.environ['HOME'],'Working/interlayer_interaction/')
sys.path.append(INTERACTION_PATH)

from make_step2 import exec_gjf
from vdw_step2 import vdw_R_step2
from vdw_step2 import detect_peaks
from utils import get_E

def init_process(args):
    # 数理モデル的に自然な定義の元のparams initリスト: not yet
    # 結晶学的に自然なパラメータへ変換: not yet
    auto_dir = args.auto_dir
    
    monomer_name = args.monomer_name
    
    os.makedirs(auto_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)

    def get_init_para_csv(auto_dir,monomer_name,params_dict):##todo
        a_ = params_dict['a']; b_ = params_dict['b']; theta = params_dict['theta']
        init_para_list=[]
        init_params_csv = os.path.join(auto_dir, 'step2_para_init_params.csv')
        A1=0;A2=0
        phi=math.atan2(b_,a_)
        R3_list=[np.round(R3,1) for R3 in np.linspace(-np.round(4,1),np.round(4,1),int(np.round(np.round(8,1)/0.4))+1)]
        R4_list=[np.round(R4,1) for R4 in np.linspace(-np.round(4,1),np.round(4,1),int(np.round(np.round(8,1)/0.4))+1)]
        S_list=[]##Sには-a*b Sの極大がa*bの極小
        for R3 in tqdm(R3_list):
            S1_list=[]
            for R4 in R4_list:
                t1_clps=vdw_R_step2(A1,A2,theta,phi,R4,'t1',monomer_name)
                t2_clps=vdw_R_step2(A1,A2,theta,-phi,R4-R3,'t2',monomer_name)
                a1=2*t1_clps*math.cos(phi)
                b1=2*t1_clps*math.sin(phi)
                a2=2*t2_clps*math.cos(phi)
                b2=2*t2_clps*math.sin(phi)
                a=max(a1,a2)
                b=max(b1,b2)
                S1_list.append(-a*b)
            S_list.append(S1_list)
        
        xyz=[]
        ma=detect_peaks(S_list, filter_size=7).mask###このfilterとorderの調整　-Rcの極大を探す
        for i  in range(len(R3_list)):
            for j in range(len(R4_list)):
                if str(ma[i][j])=='False':
                    if (R3_list[i]>-1.0) and (R4_list[j]>-1.0):
                        xyz.append([R3_list[i],R4_list[j],-S_list[i][j]])##Rcの極小値とRa,Rbを出力
                else:
                    continue
        if len(xyz)>0:
            for i in range(len(xyz)):
                init_para_list.append([a_,b_,theta,xyz[i][0],xyz[i][1],'NotYet'])

        df_init_params = pd.DataFrame(np.array(init_para_list),columns = ['a','b','theta','R3','R4','status'])##いじる
        df_init_params.to_csv(init_params_csv,index=False)
    
    params_dict_para={'a':7.2,'b':6.0,'theta':25}##########ここは入力or読み取り
    get_init_para_csv(auto_dir,monomer_name,params_dict_para)
    
    auto_csv_path = os.path.join(auto_dir,'step2_para.csv')
    if not os.path.exists(auto_csv_path):        
        df_E = pd.DataFrame(columns = ['R3','R4','a','b','theta','E','E_p1','E_p2','E_t1','E_t2','machine_type','status','file_name'])##いじる
    else:
        df_E = pd.read_csv(auto_csv_path)
        df_E = df_E[df_E['status']!='InProgress']
    df_E.to_csv(auto_csv_path,index=False)

    df_init=pd.read_csv(os.path.join(auto_dir,'step2_para_init_params.csv'))
    df_init['status']='NotYet'
    df_init.to_csv(os.path.join(auto_dir,'step2_para_init_params.csv'),index=False)

def main_process(args):
    os.chdir(os.path.join(args.auto_dir,'gaussian'))
    isOver = False
    while not(isOver):
        #check
        isOver = listen(args)
        time.sleep(1)

def listen(args):
    auto_dir = args.auto_dir
    monomer_name = args.monomer_name
    num_nodes = args.num_nodes
    isTest = args.isTest
    ##isInterlayer =args.isInterlayer
    #### TODO
    fixed_param_keys = ['a','b','theta']
    opt_param_keys = ['R3','R4']

    auto_csv = os.path.join(auto_dir,'step2_para.csv')
    df_E = pd.read_csv(auto_csv)
    df_queue = df_E.loc[df_E['status']=='InProgress',['machine_type','file_name']]
    machine_type_list = df_queue['machine_type'].values.tolist()
    len_queue = len(df_queue)
    maxnum_machine2 = 3#num_nodes/2 if num_nodes%2==0 else (num_nodes+1)/2##適宜変える
    
    for idx,row in zip(df_queue.index,df_queue.values):
        machine_type,file_name = row
        log_filepath = os.path.join(*[auto_dir,'gaussian',file_name])
        if not(os.path.exists(log_filepath)):#logファイルが生成される直前だとまずいので
            continue
        E_list=get_E(log_filepath)
        if len(E_list)!=4:##エネルギー的に等価なものを考える
            continue
        else:##エネルギーの内訳全般
            len_queue-=1;machine_type_list.remove(machine_type)
            Ep1=float(E_list[0]);Ep2=float(E_list[1]);Et1=float(E_list[2]);Et2=float(E_list[3])##ここも計算する分子数に合わせて調整##p1,p2,t1,t2の順にファイル作成
            E = 2*(Ep1+Ep2+Et1+Et2)
            #### TODO
            df_E.loc[idx, ['E_t1','E_t2','E_p1','E_p2','E','status']] = [Et1,Et2,Ep1,Ep2,E,'Done']
            df_E.to_csv(auto_csv,index=False)
            break#2つ同時に計算終わったりしたらまずいので一個で切る
    isAvailable = len_queue < num_nodes 
    machine2IsFull = machine_type_list.count(2) >= maxnum_machine2
    machine_type = 1 if machine2IsFull else 2
    if isAvailable:
        params_dict = get_params_dict(auto_dir,num_nodes, fixed_param_keys, opt_param_keys)##['a','b','theta','R3','R4']
        if len(params_dict)!=0:#終わりがまだ見えないなら
            alreadyCalculated = check_calc_status(auto_dir,params_dict)
            if not(alreadyCalculated):
                #### TODO
                file_name = exec_gjf(auto_dir, monomer_name,{**params_dict,'cx':0,'cy':0,'cz':0,'A1':0.,'A2':0.}, machine_type,isInterlayer=False,isTest=isTest)##paramsdictとか
                df_newline = pd.Series({**params_dict,'E':0.,'E_p1':0.,'E_p2':0.,'E_t1':0.,'E_t2':0.,'machine_type':machine_type,'status':'InProgress','file_name':file_name})
                df_E=df_E.append(df_newline,ignore_index=True)
                df_E.to_csv(auto_csv,index=False)
    
    init_params_csv=os.path.join(auto_dir, 'step2_para_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params_done = filter_df(df_init_params,{'status':'Done'})
    isOver = True if len(df_init_params_done)==len(df_init_params) else False
    return isOver

def check_calc_status(auto_dir,params_dict):
    df_E= pd.read_csv(os.path.join(auto_dir,'step2_para.csv'))
    if len(df_E)==0:
        return False
    df_E_filtered = filter_df(df_E, params_dict)
    df_E_filtered = df_E_filtered.reset_index(drop=True)
    try:
        status = get_values_from_df(df_E_filtered,0,'status')
        return status=='Done'
    except KeyError:
        return False

def get_params_dict(auto_dir, num_nodes, fixed_param_keys, opt_param_keys):##['a','b','theta','R3','R4']を出力
    """
    前提:
        step2_para_init_params.csvとstep2_para.csvがauto_dirの下にある
    """
    init_params_csv=os.path.join(auto_dir, 'step2_para_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step2_para.csv'))
    df_init_params_inprogress = df_init_params[df_init_params['status']=='InProgress']
    #fixed_param_keys = ['a','b','theta']     opt_param_keys = ['R3','R4']


    #最初の立ち上がり時
    if len(df_init_params_inprogress) < num_nodes:
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params,index,'status','InProgress')
            df_init_params.to_csv(init_params_csv,index=False)
            params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
            return params_dict
    for index in df_init_params.index:
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
        fixed_params_dict = df_init_params.loc[index,fixed_param_keys].to_dict()
        ### TODO
        isDone, opt_params_dict = get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict)##TF cx cy czを出力
        if isDone:
            # df_init_paramsのstatusをupdate
            df_init_params = update_value_in_df(df_init_params,index,'status','Done')
            if np.max(df_init_params.index) < index+1:
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params,index+1,'status')
            df_init_params.to_csv(init_params_csv,index=False)
            
            if status=='NotYet':                
                opt_params_dict = get_values_from_df(df_init_params,index+1,opt_param_keys)
                df_init_params = update_value_in_df(df_init_params,index+1,'status','InProgress')
                df_init_params.to_csv(init_params_csv,index=False)
                return {**fixed_params_dict,**opt_params_dict}
            else:
                continue

        else:
            df_inprogress = filter_df(df_cur, {**fixed_params_dict,**opt_params_dict,'status':'InProgress'})
            if len(df_inprogress)>=1:
                continue
            return {**fixed_params_dict,**opt_params_dict}
    return {}
        
def get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict):
    df_val = filter_df(df_cur, fixed_params_dict)
    R3_init_prev = init_params_dict['R3']; R4_init_prev = init_params_dict['R4']
    
    while True:######todo
        E_list=[];r3r4_list=[]
        for R3 in [R3_init_prev-0.2,R3_init_prev,R3_init_prev+0.2]:
            for R4 in [R4_init_prev-0.2,R4_init_prev,R4_init_prev+0.2]:
                R3 = np.round(R3,1);R4 = np.round(R4,1)
                df_val_r3r4 = df_val[
                (df_val['R3']==R3)&(df_val['R4']==R4)
                &(df_val['status']=='Done')
                   ]
                if len(df_val_r3r4)==0:
                     return False,{'R3':R3,'R4':R4}
                r3r4_list.append([R3,R4]);E_list.append(df_val_r3r4['E'].values[0])
        R3_init,R4_init = r3r4_list[np.argmin(np.array(E_list))]
        if R3_init==R3_init_prev and R4_init==R4_init_prev:
            return True,{'R3':R3_init,'R4':R4_init}
        else:
             R3_init_prev=R4_init;R4_init_prev=R4_init
    
def get_values_from_df(df,index,key):
    return df.loc[index,key]

def update_value_in_df(df,index,key,value):
    df.loc[index,key]=value
    return df

def filter_df(df, dict_filter):
    query = []
    for k, v in dict_filter.items():
        if type(v)==str:
            query.append('{} == "{}"'.format(k,v))
        else:
            query.append('{} == {}'.format(k,v))
    df_filtered = df.query(' and '.join(query))
    return df_filtered

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--init',action='store_true')
    parser.add_argument('--isTest',action='store_true')
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes gaussian, gaussview and csv')
    parser.add_argument('--monomer-name',type=str,help='monomer name')
    parser.add_argument('--num-nodes',type=int,help='num nodes')
    
    args = parser.parse_args()

    if args.init:
        print("----initial process----")
        init_process(args)
    
    print("----main process----")
    main_process(args)
    print("----finish process----")
