import numpy as np
import torch
import torch.nn as nn
import datetime
import pytz
import os
import lifelines.utils.concordance as LUC
import torch.nn.functional as F
from torch.nn import init

from tqdm import tqdm
from torch_geometric.nn import GATConv as GATConv_v1
from torch_geometric.nn import GATv2Conv as GATConv

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def non_decay_filter(model):

    no_decay = list()
    decay = list()

    for m in model.modules():
        if isinstance(m, nn.Linear):
            decay.append(m.weight)
            if m.bias != None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            no_decay.append(m.weight)
            if m.bias != None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.LayerNorm):
            no_decay.append(m.weight)
            if m.bias != None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.PReLU):
            no_decay.append(m.weight)
        elif isinstance(m, GATConv):
            decay.append(m.att)
            if m.bias != None:
                no_decay.append(m.bias)
            no_decay.append(m.position_bias)
        elif isinstance(m, GATConv_v1):
            decay.append(m.att_l)
            decay.append(m.att_r)
            if m.bias != None:
                no_decay.append(m.bias)
            no_decay.append(m.position_bias)
            no_decay.append(m.angle_bias)
            decay.append(m.att_edge_attr_pos)
            decay.append(m.att_edge_attr_angle)

    model_parameter_groups = [dict(params=decay), dict(params=no_decay, weight_decay=0.0)]

    return model_parameter_groups

def metadata_list_generation(DatasetType, Trainlist, Metadata):

    Train_survivallist = []
    Train_censorlist = []
    Train_stagelist = []
    exclude_list = []

    if "LIHC" in DatasetType:

        with tqdm(total=len(Trainlist)) as tbar:
            for idx in range(len(Trainlist)):

                item = '-'.join(Trainlist[idx].split('/')[-1].split('.pt')[0].split('_')[0].split('-')[0:3])

                Match_item = Metadata[Metadata["case_submitter_id"] == item]
                if Match_item.shape[0] != 0:
                    if Match_item['vital_status'].tolist()[0] == "Alive":
                        if '--' not in Match_item['days_to_last_follow_up'].tolist()[0]:
                            if '--' not in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                Train_censorlist.append(0)
                                Train_survivallist.append(
                                    int(float(Match_item['days_to_last_follow_up'].tolist()[0])))
                                if ('IV' or 'X') in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(4)
                                elif "III" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(3)
                                elif "II" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(2)
                                elif "I" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(1)
                                else:
                                    Train_stagelist.append(0)
                            else:
                                exclude_list.append(idx)
                        else:
                            exclude_list.append(idx)
                    else:
                        if '--' not in Match_item['days_to_death'].tolist()[0]:
                            if '--' not in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                Train_censorlist.append(1)
                                Train_survivallist.append(int(float(Match_item['days_to_death'].tolist()[0])))

                                if ('IV' or 'X') in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(4)
                                elif "III" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(3)
                                elif "II" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(2)
                                elif "I" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(1)
                                else:
                                    Train_stagelist.append(0)
                            else:
                                exclude_list.append(idx)
                        else:
                            exclude_list.append(idx)
                else:
                    exclude_list.append(idx)

        _ = [Trainlist.pop(idx_item - c) for c, idx_item in enumerate(exclude_list)]


    elif "ES2" in DatasetType:

        with tqdm(total=len(Trainlist)) as tbar:
            for idx in range(len(Trainlist)):

                item = '-'.join(Trainlist[idx].split('/')[-1].split('.pt')[0].split('_')[0].split('-')[0:3])
                Match_item = Metadata[Metadata["id"] == item]
                if Match_item.shape[0] != 0:
                    # if Match_item['death'].tolist()[0] == "0":
                    if str('0') in str(Match_item['death'].tolist()[0]):
                        # if '--' not in Match_item['OS'].tolist()[0]:
                        # if '--' not in Match_item['MVI'].tolist()[0]:
                        Train_censorlist.append(0)
                        Train_survivallist.append(float(Match_item['OS'].tolist()[0]))
                        if str('1') in str(Match_item['MVI'].tolist()[0]):
                            Train_stagelist.append(1)
                        else:
                            Train_stagelist.append(2)
                    # else:
                    # exclude_list.append(idx)
                    # else:
                    # exclude_list.append(idx)
                    else:
                        # if '--' not in Match_item['OS'].tolist()[0]:
                        # if '--' not in Match_item['MVI'].tolist()[0]:
                        Train_censorlist.append(1)
                        Train_survivallist.append(float(Match_item['OS'].tolist()[0]))

                        if str('1') in str(Match_item['MVI'].tolist()[0]):
                            Train_stagelist.append(1)
                        else:
                            Train_stagelist.append(2)
                    # else:
                    # exclude_list.append(idx)
                    # else:
                    # exclude_list.append(idx)
                else:
                    exclude_list.append(idx)

        _ = [Trainlist.pop(idx_item - c) for c, idx_item in enumerate(exclude_list)]

    elif "ES" in DatasetType:

        with tqdm(total=len(Trainlist)) as tbar:
            for idx in range(len(Trainlist)):

                item = '-'.join(Trainlist[idx].split('/')[-1].split('.pt')[0].split('_')[0].split('-')[0:3])
                Match_item = Metadata[Metadata["id"] == item]
                if Match_item.shape[0] != 0:
                    #if Match_item['death'].tolist()[0] == "0":
                    if str('0') in str(Match_item['death'].tolist()[0]):
                        #if '--' not in Match_item['OS'].tolist()[0]:
                            #if '--' not in Match_item['MVI'].tolist()[0]:
                                Train_censorlist.append(0)
                                Train_survivallist.append(float(Match_item['OS'].tolist()[0]))
                                if str('1') in str(Match_item['MVI'].tolist()[0]):
                                    Train_stagelist.append(1)
                                else:
                                    Train_stagelist.append(2)
                            #else:
                                #exclude_list.append(idx)
                        #else:
                            #exclude_list.append(idx)
                    else:
                        #if '--' not in Match_item['OS'].tolist()[0]:
                            #if '--' not in Match_item['MVI'].tolist()[0]:
                                Train_censorlist.append(1)
                                Train_survivallist.append(float(Match_item['OS'].tolist()[0]))

                                if str('1') in str(Match_item['MVI'].tolist()[0]):
                                    Train_stagelist.append(1)
                                else:
                                    Train_stagelist.append(2)
                            #else:
                                #exclude_list.append(idx)
                        #else:
                            #exclude_list.append(idx)
                else:
                    exclude_list.append(idx)

        _ = [Trainlist.pop(idx_item - c) for c, idx_item in enumerate(exclude_list)]
        #print(Trainlist)
        #print(len(Train_survivallist))
        #print(len(Train_censorlist))
        #print(len(Train_stagelist))
    return Trainlist, Train_survivallist, Train_censorlist, Train_stagelist

def train_test_split(Trainlist, Metadata, DatasetType, TrainRoot, Fi, Analyze_flag=False):

    Trainlist, Train_survivallist, Train_censorlist, Train_stagelist = metadata_list_generation(DatasetType, Trainlist, Metadata)
    Train_not_eventlist = np.where(np.array(Train_censorlist) == 0)[0]
    Train_not_eventlist = [item for c, item in enumerate(Train_not_eventlist)]
    Train_eventlist = np.where(np.array(Train_censorlist) == 1)[0]
    Train_eventlist = [item for c, item in enumerate(Train_eventlist)]

    Wholelist = [os.path.join(TrainRoot, item) for item in Trainlist]

    Train_noncensor_survivallist = np.array(Train_survivallist)[Train_not_eventlist[0:int(len(Train_not_eventlist) * 0.8)]]
    Train_noncensor_survivallist = Train_noncensor_survivallist.tolist()
    Train_censor_survivallist = np.array(Train_survivallist)[Train_eventlist[0:int(len(Train_eventlist) * 0.8)]]
    Train_censor_survivallist = Train_censor_survivallist.tolist()

    Train_noncensor_stagelist = np.array(Train_stagelist)[Train_not_eventlist[0:int(len(Train_not_eventlist) * 0.8)]]
    Train_noncensor_stagelist = Train_noncensor_stagelist.tolist()
    Train_censor_stagelist = np.array(Train_stagelist)[Train_eventlist[0:int(len(Train_eventlist) * 0.8)]]
    Train_censor_stagelist = Train_censor_stagelist.tolist()

    Train_noncensor_censorlist = np.array(Train_censorlist)[Train_not_eventlist[0:int(len(Train_not_eventlist) * 0.8)]]
    Train_noncensor_censorlist = Train_noncensor_censorlist.tolist()
    Train_censor_censorlist = np.array(Train_censorlist)[Train_eventlist[0:int(len(Train_eventlist) * 0.8)]]
    Train_censor_censorlist = Train_censor_censorlist.tolist()

    Trainlist_noncensor = np.array(Wholelist)[Train_not_eventlist[0:int(len(Train_not_eventlist) * 0.8)]]
    Trainlist_noncensor = Trainlist_noncensor.tolist()
    Trainlist_censor = np.array(Wholelist)[Train_eventlist[0:int(len(Train_eventlist) * 0.8)]]
    Trainlist_censor = Trainlist_censor.tolist()

    Test_noncensor_survivallist = np.array(Train_survivallist)[Train_not_eventlist[int(len(Train_not_eventlist) * 0.8):len(Train_not_eventlist)]]
    Test_censor_survivallist = np.array(Train_survivallist)[Train_eventlist[int(len(Train_eventlist) * 0.8):len(Train_eventlist)]]
    Test_survivallist = Test_noncensor_survivallist.tolist() + Test_censor_survivallist.tolist()

    Test_noncensor_stagelist = np.array(Train_stagelist)[Train_not_eventlist[int(len(Train_not_eventlist) * 0.8):len(Train_not_eventlist)]]
    Test_censor_stagelist = np.array(Train_stagelist)[Train_eventlist[int(len(Train_eventlist) * 0.8):len(Train_eventlist)]]
    Test_stagelist = Test_noncensor_stagelist.tolist() + Test_censor_stagelist.tolist()

    Test_noncensor_censorlist = np.array(Train_censorlist)[Train_not_eventlist[int(len(Train_not_eventlist) * 0.8):len(Train_not_eventlist)]]
    Test_censor_censorlist = np.array(Train_censorlist)[Train_eventlist[int(len(Train_eventlist) * 0.8):len(Train_eventlist)]]
    Test_censorlist = Test_noncensor_censorlist.tolist() + Test_censor_censorlist.tolist()

    Testlist_noncensor = np.array(Wholelist)[Train_not_eventlist[int(len(Train_not_eventlist) * 0.8):len(Train_not_eventlist)]]
    Testlist_censor = np.array(Wholelist)[Train_eventlist[int(len(Train_eventlist) * 0.8):len(Train_eventlist)]]
    Testlist = Testlist_noncensor.tolist() + Testlist_censor.tolist()

    TrainFF = np.array(
        Trainlist_noncensor[0:Fi * int(len(Trainlist_noncensor) / 5)] + Trainlist_noncensor[(Fi + 1) * int(len(Trainlist_noncensor) / 5):len(Trainlist_noncensor)]\
        + Trainlist_censor[0:Fi * int(len(Trainlist_censor) / 5)] + Trainlist_censor[(Fi + 1) * int(len(Trainlist_censor) / 5):len(Trainlist_censor)])

    TrainFF_survivallist = \
        Train_noncensor_survivallist[0:Fi * int(len(Train_noncensor_survivallist) / 5)] + Train_noncensor_survivallist[(Fi + 1) * int(len(Train_noncensor_survivallist) / 5):len(Train_noncensor_survivallist)]\
        + Train_censor_survivallist[0:Fi * int(len(Train_censor_survivallist) / 5)] + Train_censor_survivallist[(Fi + 1) * int(len(Train_censor_survivallist) / 5):len(Train_censor_survivallist)]

    TrainFF_stagelist = \
        Train_noncensor_stagelist[0:Fi * int(len(Train_noncensor_stagelist) / 5)] + Train_noncensor_stagelist[(Fi + 1) * int(len(Train_noncensor_stagelist) / 5):len(Train_noncensor_stagelist)]\
        + Train_censor_stagelist[0:Fi * int(len(Train_censor_stagelist) / 5)] + Train_censor_stagelist[(Fi + 1) * int(len(Train_censor_stagelist) / 5):len(Train_censor_stagelist)]

    TrainFF_censorlist = \
        Train_noncensor_censorlist[0:Fi * int(len(Train_noncensor_censorlist) / 5)] + Train_noncensor_censorlist[(Fi + 1) * int(len(Train_noncensor_censorlist) / 5):len(Train_noncensor_censorlist)]\
        + Train_censor_censorlist[0:Fi * int(len(Train_censor_censorlist) / 5)] + Train_censor_censorlist[(Fi + 1) * int(len(Train_censor_censorlist) / 5):len(Train_censor_censorlist)]

    ValidFF = np.array(
        Trainlist_noncensor[Fi * int(len(Trainlist_noncensor) / 5):(Fi + 1) * int(len(Trainlist_noncensor) / 5)]
        + Trainlist_censor[Fi * int(len(Trainlist_censor) / 5):(Fi + 1) * int(len(Trainlist_censor) / 5)])

    ValidFF_survivallist = Train_noncensor_survivallist[Fi * int(len(Train_noncensor_survivallist) / 5):(Fi + 1) * int(
        len(Train_noncensor_survivallist) / 5)] \
                           + Train_censor_survivallist[Fi * int(len(Train_censor_survivallist) / 5):(Fi + 1) * int(
        len(Train_censor_survivallist) / 5)]

    ValidFF_stagelist = Train_noncensor_stagelist[
                        Fi * int(len(Train_noncensor_stagelist) / 5):(Fi + 1) * int(len(Train_noncensor_stagelist) / 5)] \
                        + Train_censor_stagelist[
                          Fi * int(len(Train_censor_stagelist) / 5):(Fi + 1) * int(len(Train_censor_stagelist) / 5)]

    ValidFF_censorlist = Train_noncensor_censorlist[
                         Fi * int(len(Train_noncensor_censorlist) / 5):(Fi + 1) * int(len(Train_noncensor_censorlist) / 5)] \
                         + Train_censor_censorlist[
                           Fi * int(len(Train_censor_censorlist) / 5):(Fi + 1) * int(len(Train_censor_censorlist) / 5)]

    TrainFF_set = (TrainFF, TrainFF_survivallist, TrainFF_censorlist, TrainFF_stagelist)
    ValidFF_set = (ValidFF, ValidFF_survivallist, ValidFF_censorlist, ValidFF_stagelist)
    Test_set = (Testlist, Test_survivallist, Test_censorlist, Test_stagelist)

    if Analyze_flag == True:
        Test_set = (np.array(TrainFF.tolist() + ValidFF.tolist() + Testlist),
                    TrainFF_survivallist + ValidFF_survivallist + Test_survivallist,
                    TrainFF_censorlist + ValidFF_censorlist + Test_censorlist,
                    TrainFF_stagelist + ValidFF_stagelist + Test_stagelist)
        return Test_set

    else:
        return TrainFF_set, ValidFF_set, Test_set

def makecheckpoint_dir_graph(Argument):
    todaydata = datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d_%H_%M_%S")

    checkpoint_dir = os.path.join('./results/', Argument.DatasetType, Argument.model)
    checkpoint_dir = os.path.join(checkpoint_dir, todaydata)
    if os.path.exists(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)

    figure_dir = os.path.join(checkpoint_dir, "Figure")
    if os.path.exists(figure_dir) is False:
        os.mkdir(figure_dir)

    return checkpoint_dir, figure_dir

class coxph_loss(torch.nn.Module):

    def __init__(self):
        super(coxph_loss, self).__init__()

    def forward(self, risk, phase, censors):
        #print('risk',risk)
        #print('censors',censors)
        #riskmax = risk
        riskmax = F.normalize(risk, p=2, dim=0)
        #print('riskmax',riskmax)
        log_risk = torch.log((torch.cumsum(torch.exp(riskmax), dim=0)))
        #uncensored_likelihood = torch.cumsum(torch.exp(riskmax), dim=0)
        #print('log_risk',log_risk)
        uncensored_likelihood = torch.add(riskmax, -log_risk)
        #print('uncensored_likelihood',uncensored_likelihood)
        resize_censors = censors.resize_(uncensored_likelihood.size()[0], 1)
        censored_likelihood = torch.mul(uncensored_likelihood, resize_censors)
        loss = -torch.sum(censored_likelihood) / float(censors.nonzero().size(0))
        #loss = -torch.sum(censored_likelihood) / float(censors.size(0))

        return loss

class coxph_loss2(torch.nn.Module):

    def __init__(self):
        super(coxph_loss, self).__init__()

    def forward(self, risk, phase, censors):
        #riskmax = risk
        riskmax = F.normalize(risk, p=2, dim=0)
        # riskmax=risk
        srisk = 1 - riskmax
        # print(srisk)
        decensors = 1 - censors
        s = torch.cumsum(srisk, dim=0)
        st = s - srisk
        cenloss = torch.add(torch.mul(censors, torch.log(s)), torch.mul(decensors, torch.log(s)))
        cenloss2 = torch.add(cenloss, torch.mul(decensors, torch.log(riskmax)))
        loss = torch.sum(cenloss2)
        #  / float(censors.size(0))
        return loss

class coxph_loss3(torch.nn.Module):

    def __init__(self):
        super(coxph_loss, self).__init__()

    def forward(self, risk, phase, censors):
        riskmax = risk
        #riskmax = F.normalize(risk, p=2, dim=0)
        s = torch.log(1 - riskmax)
        S = torch.cumsum(s, dim=0)
        stx = torch.exp(S)
        S2 = torch.log(1 - stx)
        lo = torch.add(torch.mul(censors, S), torch.mul(1 - censors, S2))
        loss = -torch.sum(lo)

        return loss

class coxph_loss4(torch.nn.Module):

    def __init__(self):
        super(coxph_loss, self).__init__()

    def forward(self, risk, phase, censors,S=None,alpha=0.5,eps=1e-7):
        batch_size=len(risk)
        c=censors.view(batch_size,1).float()
        if S is None:
            S=torch.cumprod(1-risk,dim=1)
        S_padded=torch.cat([torch.ones_like(c),S],1)
        uncensored_loss=-c*(torch.log(S_padded.clamp(min=eps))+torch.log(risk.clamp(min=eps)))
        censored_loss=-(1-c)*torch.log(S_padded.clamp(min=eps))
        neg_l=censored_loss+uncensored_loss
        loss=(1-alpha)*neg_l+alpha*uncensored_loss
        loss=loss.mean()

        return loss


def accuracytest(survivals, risk, censors):
    survlist = []
    risklist = []
    censorlist = []

    for riskval in risk:
        risklist.append(riskval.cpu().detach().item())

    for censorval in censors:
        censorlist.append(censorval.cpu().detach().item())

    for surval in survivals:
        survlist.append(surval.cpu().detach().item())
    #print(len(survlist))
    #print(len(risklist))
    #print(len(censorlist))
    C_value = LUC.concordance_index(survlist, -np.exp(risklist), censorlist)


    return C_value

def TrainValid_path(DatasetType):
    Pretrain_root = 0

    if DatasetType == "BORAME":
        Pretrain_root = "./Sample_data_for_demo/Graph_test/TCGA/KIRC/superpatch5/"
    elif DatasetType == "LIHC":
        Pretrain_root = "./Sample_data_for_demo/Graph_test/TCGA/KIRC/superpatch6/"
    elif DatasetType == "sichuan":
        Pretrain_root = "./Sample_data_for_demo/Graph_test/TCGA/KIRC/superpatch8/"
    elif DatasetType == "jiaxin":
        Pretrain_root = "./Sample_data_for_demo/Graph_test/TCGA/KIRC/superpatch7/"
    elif DatasetType == "binzhou":
        Pretrain_root = "./Sample_data_for_demo/Graph_test/TCGA/KIRC/superpatch9/"
    elif DatasetType == "TCGA":
        Pretrain_root = "./Sample_data_for_demo/Graph_test/TCGA/KIRC/superpatch3/"
    elif DatasetType == "TCGA2":
        Pretrain_root = "./Sample_data_for_demo/Graph_test/TCGA/KIRC/superpatch/"
    elif DatasetType == "TCGA-incno":
        #Pretrain_root = "./Sample_data_for_demo/Graphdata/KIRC/"
        Pretrain_root = "./Sample_data_for_demo/Graph_test/TCGA/KIRC/superpatch2/"
    elif DatasetType == "TCGA-effno":
        #Pretrain_root = "./Sample_data_for_demo/Graphdata/KIRC/"
        Pretrain_root = "./Sample_data_for_demo/Graph_test/TCGA/KIRC/superpatch4/"
    elif DatasetType == "NLST":
        Pretrain_root = "./Sample_data_for_demo/Graphdata/NLST/"

    return Pretrain_root

def cox_sort(out, tempsurvival, tempphase, tempmeta, tempstage, tempID,
             EpochSurv, EpochPhase, EpochRisk, EpochStage, EpochID):

    sort_idx = torch.argsort(tempsurvival, descending=True)
    updated_feature_list = []

    risklist = out[sort_idx]
    tempsurvival = tempsurvival[sort_idx]
    tempphase = tempphase[sort_idx]
    tempmeta = tempmeta[sort_idx]
    for idx in sort_idx.cpu().detach().tolist():
        EpochID.append(tempID[idx])
    tempstage = tempstage[sort_idx]

    risklist = risklist.to(out.device)
    tempsurvival = tempsurvival.to(out.device)
    tempphase = tempphase.to(out.device)
    tempmeta = tempmeta.to(out.device)

    for riskval, survivalval, phaseval, stageval, metaval in zip(risklist, tempsurvival,
                                                                 tempphase, tempstage,
                                                                 tempmeta):
        EpochSurv.append(survivalval.cpu().detach().item())
        EpochPhase.append(phaseval.cpu().detach().item())
        EpochRisk.append(riskval.cpu().detach().item())
        EpochStage.append(stageval.cpu().detach().item())

    return risklist, tempsurvival, tempphase, tempmeta, EpochSurv, EpochPhase, EpochRisk, EpochStage
