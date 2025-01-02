



import os
import sys
import numpy as np
import torch
sys.path.append(os.path.abspath('/home/mimic/MIMIC_subset/MIMIC_subset'))
from dataset.update_ECGdataset import get_ECG_datasets,get_data_loader,my_collate
np.set_printoptions(threshold=sys.maxsize)
# from dataset.ECG_dataset import get_ECG_datasets,get_data_loader,my_collate

from torch.utils.data import DataLoader
torch.set_printoptions(threshold=torch.inf, linewidth=1000)
from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()

train_ds,val_ds,test_ds=get_ECG_datasets(args)
# print(train[-1])
# print(train[-2])
train_dl,val_dl=get_data_loader(batch_size=32)
# for batch_idx, (x,y) in enumerate(val_dl):
#     print(x)
#     with open("train_dataset1.txt", "w") as f:
#         f.write(repr(x)) 
#         break

# with open("dataset2.txt", "w") as f:
#     f.write(repr(val_dl[-2]))

def getStat(val_ds):
    # print(len(train_ds))
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)
    # train_dl=torch.utils.data.DataLoader(
        # train_ds, batch_size=16, shuffle=False, num_workers=0,
        # pin_memory=True,collate_fn=my_collate)
    mean=torch.zeros(12)
    std=torch.zeros(12)
    i=0
    for _,x,_ in val_dl:
        i+=1
        # x = torch.from_numpy(x)
        # x=torch.transpose(x,2,1)
        print(f'getstat ecg {x.shape}') #[16,12,4096](frequency)/[bs,257,17,12]
        for d in range(12):
            # a=x[:,:,:,d]#[12,257,17]
            # print(f'a shape {a.shape}')
            # x = torch.from_numpy(x)
            # x=torch.transpose(x,2,1)
            # print(x.shape)
            mean[d]+=x[:,d,:].mean()
            std[d]+=x[:,d,:].std()
            # mean[d]+=x[:,:,:,d].mean()
            # std[d]+=x[:,:,:,d].std()
            # print(d)
        # print(i)
    mean.div_(len(val_dl))
    std.div_(len(val_dl))
    # return list(mean.numpy()),list(std.numpy())
    return mean.tolist(), std.tolist()


# mean,std=getStat(val_ds) 
# print(f'val mean {mean}')
# print(f'val std {std}')
#-fequency ------------------------
# train mean [0.0010015363804996014, 0.0036762908566743135, 0.0035229665227234364, 0.003404112532734871, 0.0034496726002544165, 0.003417636500671506, 0.0034184211399406195, 0.0034051071852445602, 0.00337942223995924, 0.0034143230877816677, 0.003411758691072464, 0.0033634386491030455]
# train std [0.0038484581746160984, 0.012485950253903866, 0.011201074346899986, 0.010804145596921444, 0.010862342081964016, 0.01069234311580658, 0.01059968676418066, 0.010482105426490307, 0.010425384156405926, 0.01052883081138134, 0.010451158508658409, 0.010305781848728657]
# val mean [0.002673633163794875, 0.0025307531468570232, 0.002691473113372922, 0.0022021227050572634, 0.0022452438715845346, 0.0023844053503125906, 0.003116462379693985, 0.004411369562149048, 0.004606869071722031, 0.00413723336532712, 0.0036658409517258406, 0.0031537378672510386]
# val std [0.007418525870889425, 0.007158512249588966, 0.008282571099698544, 0.0059426347725093365, 0.006800748407840729, 0.007019363809376955, 0.010448685847222805, 0.01500698458403349, 0.015392810106277466, 0.012730532325804234, 0.010609139688313007, 0.009342469274997711]

#------------temporal-------------

#val mean [0.02472313866019249, 0.015134230256080627, -0.008734860457479954, -0.01945239119231701, 0.0027260903734713793, 0.016174761578440666, -0.019995039328932762, -0.00462614931166172, -0.003113960847258568, 0.010996637865900993, 0.020355889573693275, 0.02063339576125145]
# val std [0.16145117580890656, 0.1541784554719925, 0.1643943190574646, 0.13462091982364655, 0.13751928508281708, 0.1440832018852234, 0.19534701108932495, 0.26101362705230713, 0.2725179195404053, 0.2411022186279297, 0.2139832228422165, 0.18245722353458405]

# train mean [0.02163420245051384, 0.012705769389867783, -0.007751110941171646, -0.016836699098348618, 0.0019329616334289312, 0.014118772000074387, -0.01788940653204918, -0.0035780423786491156, -0.00025098101468756795, 0.013947268947958946, 0.019551198929548264, 0.01937594637274742]
# train std [0.1569816619157791, 0.15234516561031342, 0.16046899557113647, 0.13147377967834473, 0.13560453057289124, 0.14005501568317413, 0.18976102769374847, 0.2568478286266327, 0.2705800533294678, 0.24009983241558075, 0.2132350653409958, 0.18305405974388123]

#without nan------------------------
# train mean [0.021983036771416664, 0.013289721682667732, -0.007492052856832743, -0.0172954760491848, 0.002315341029316187, 0.014129746705293655, -0.018112635239958763, -0.0034318040125072002, 0.00021024956367909908, 0.014880170114338398, 0.02024325355887413, 0.01977406069636345]
# train std [0.15690241754055023, 0.15237219631671906, 0.15942928194999695, 0.13163667917251587, 0.13473273813724518, 0.13911446928977966, 0.18939709663391113, 0.25870004296302795, 0.2708594501018524, 0.2404349446296692, 0.21415464580059052, 0.18362797796726227]

# val mean [0.02482433430850506, 0.015947148203849792, -0.007936287671327591, -0.019916854798793793, 0.0034988929983228445, 0.01582620106637478, -0.019720222800970078, -0.004102816805243492, -0.0021764931734651327, 0.012111539952456951, 0.021004941314458847, 0.02135436423122883]
# val std [0.1607745736837387, 0.15428286790847778, 0.16370441019535065, 0.1342308074235916, 0.13679346442222595, 0.14322689175605774, 0.1934480518102646, 0.2608761787414551, 0.27516594529151917, 0.24000638723373413, 0.2119644731283188, 0.18333952128887177]



def get_fre_Stat(val_ds):
    # print(len(train_ds))
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)
    # train_dl=torch.utils.data.DataLoader(
        # train_ds, batch_size=16, shuffle=False, num_workers=0,
        # pin_memory=True,collate_fn=my_collate)
    mean=torch.zeros(12)
    std=torch.zeros(12)
    i=0
    for _,x,_ in val_dl:
        i+=1
        # x = torch.from_numpy(x)
        # x=torch.transpose(x,2,1)
        print(f'get_fre_stat ecg {x.shape}') #[16,257,17,12]
        for d in range(12):
            print(f'111 x {x.shape}')
            a=x[:,:,:,d]
            print(f'a shape {a.shape}')
            # x = torch.from_numpy(x)
            # x=torch.transpose(x,2,1)
            # print(x.shape)
            # mean[d]+=x[:,d,:].mean()
            # std[d]+=x[:,d,:].std()
            mean[d]+=x[:,:,:,d].mean()
            std[d]+=x[:,:,:,d].std()
            # print(d)
        # print(i)
    mean.div_(len(val_dl))
    std.div_(len(val_dl))
    # return list(mean.numpy()),list(std.numpy())
    return mean.tolist(), std.tolist()


mean,std=get_fre_Stat(val_ds) 
print(f'val mean {mean}')
print(f'val std {std}')

# val mean [0.033378660678863525, 0.03305447846651077, 0.034193579107522964, 0.031550049781799316, 0.03091508150100708, 0.02711910754442215, 0.024816736578941345, 0.02398015186190605, 0.02438199333846569, 0.02432316541671753, 0.023364150896668434, 0.021768491715192795]
# val std [0.04742216691374779, 0.0386776477098465, 0.039538607001304626, 0.03581101447343826, 0.03057497926056385, 0.024299126118421555, 0.02237706072628498, 0.022275008261203766, 0.021296387538313866, 0.020055022090673447, 0.018722841516137123, 0.017433255910873413]

# train mean [0.03296802565455437, 0.03273816406726837, 0.03355516120791435, 0.0311842393130064, 0.030348533764481544, 0.027099210768938065, 0.024589119479060173, 0.023644540458917618, 0.02395729161798954, 0.023933202028274536, 0.0230211541056633, 0.021562350913882256]
# train std [0.04679363593459129, 0.0376615896821022, 0.03844350948929787, 0.0354924313724041, 0.03010416217148304, 0.024544646963477135, 0.022233400493860245, 0.021855955943465233, 0.020977528765797615, 0.0198527779430151, 0.018527885898947716, 0.017392240464687347]
