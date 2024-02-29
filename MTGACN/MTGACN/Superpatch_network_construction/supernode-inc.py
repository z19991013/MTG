#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:10:37 2021

@author: kyungsub
"""
from torchvision.models import Inception_V3_Weights
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import openslide as osd
from torchvision import transforms
from torch_geometric.data import Data
from Superpatch_network_construction.EfficientNet import EfficientNet
# from Superpatch_network_construction.superpatch_network_construction import false_graph_filtering
from Superpatch_network_construction.test import false_graph_filtering
from skimage.filters import threshold_multiotsu
import pickle
import argparse
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from xml.etree import ElementTree as ET
import concurrent.futures
import cv2 as cv
import pandas
from torchvision.models import Inception_V3_Weights

def Prediction(model_N,images,device):
    '''
    param model_N: Neural network model suitable for Pytorch
    param test_data: Pytoch dataloader form

    return:
        name_list: The name list of the predicted image.
        preds_list: The list of predicted category.
        values_list: The list of prediction probability.
    '''

    with torch.no_grad():
        model_N.eval()

        preds_list = []
        values_list = []
        output_list=[]

        inputs = images.to(device)

        outputs = model_N(inputs)

        predict = torch.sigmoid(outputs)
        values, preds = torch.max(predict, 1)
        values_c = values.cpu().numpy()
        op=outputs.cpu().numpy()
        output_list.extend(op)
        preds_list.extend(preds.cpu().numpy())
        values_list.extend(np.round(values_c, 2))

    return  preds_list, values_list,output_list

class SurvivalImageDataset():
    """
    Target dataset has the list of images such as
    _patientID_SurvDay_Censor_TumorStage_WSIPos.tif
    """

    def __init__(self, image, x, y, transform):
        self.image = image
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len((self.image))

    def __getitem__(self, idx):
        """
        patientID, SurvivalDuration, SurvivalCensor, Stage,
        ProgressionDuration, ProgressionCensor, MetaDuration, MetaCensor
        """
        transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        image = self.image[idx]
        x = self.x[idx]
        y = self.y[idx]
        image = image.convert('RGB')
        R = transform(image)

        sample = {'image': R, 'X': torch.tensor(x), 'Y': torch.tensor(y)}

        return sample


def supernode_generation(image, model_ft, device, Argument, save_dir):
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    origin_dir = os.path.join(save_dir, 'original')
    if os.path.exists(origin_dir) is False:
        os.mkdir(origin_dir)

    superpatch_dir = os.path.join(save_dir, 'superpatch')
    if os.path.exists(superpatch_dir) is False:
        os.mkdir(superpatch_dir)

    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    threshold = Argument.threshold
    spatial_threshold = Argument.spatial_threshold

    sample = image.split('/')[-1].split('.')[0]
    sample = sample[:-23]
    #sample = sample[:-4]
    print(sample)

    image_path = image
    try:
        slideimage = osd.OpenSlide(image_path)
    except:
        print('openslide error')
        return 0
    downsampling = slideimage.level_downsamples
    if len(downsampling) > 2:
        best_downsampling_level = 2
        downsampling_factor = int(slideimage.level_downsamples[best_downsampling_level])

        # Get the image at the requested scale
        svs_native_levelimg = slideimage.read_region((0, 0), best_downsampling_level,
                                                     slideimage.level_dimensions[best_downsampling_level])
        svs_native_levelimg = svs_native_levelimg.convert('L')
        img = np.array(svs_native_levelimg)

        thresholds = threshold_multiotsu(img)
        regions = np.digitize(img, bins=thresholds)
        regions[regions == 1] = 0
        regions[regions == 2] = 1
        thresh_otsu = regions

        imagesize = Argument.imagesize
        downsampled_size = int(imagesize / downsampling_factor)
        Width = slideimage.dimensions[0]
        Height = slideimage.dimensions[1]
        num_row = int(Height / imagesize) + 1
        num_col = int(Width / imagesize) + 1
        x_list = []
        y_list = []
        feature_list = []
        x_y_list = []
        counter = 0
        inside_counter = 0
        temp_patch_list = []
        temp_x = []
        temp_y = []

        with tqdm(total=num_row * num_col) as pbar_image:
            for i in range(0, num_col):
                for j in range(0, num_row):

                    if thresh_otsu.shape[1] >= (i + 1) * downsampled_size:
                        if thresh_otsu.shape[0] >= (j + 1) * downsampled_size:
                            cut_thresh = thresh_otsu[j * downsampled_size:(j + 1) * downsampled_size,
                                         i * downsampled_size:(i + 1) * downsampled_size]
                        else:
                            cut_thresh = thresh_otsu[(j) * downsampled_size:thresh_otsu.shape[0],
                                         i * downsampled_size:(i + 1) * downsampled_size]
                    else:
                        if thresh_otsu.shape[0] >= (j + 1) * downsampled_size:
                            cut_thresh = thresh_otsu[j * downsampled_size:(j + 1) * downsampled_size,
                                         (i) * downsampled_size:thresh_otsu.shape[1]]
                        else:
                            cut_thresh = thresh_otsu[(j) * downsampled_size:thresh_otsu.shape[0],
                                         (i) * downsampled_size:thresh_otsu.shape[1]]

                    if np.mean(cut_thresh) > 0.75:
                        pbar_image.update()
                        pass
                    else:

                        filter_location = (i * imagesize, j * imagesize)
                        level = 0
                        patch_size = (imagesize, imagesize)
                        location = (filter_location[0], filter_location[1])

                        CutImage = slideimage.read_region(location, level, patch_size)

                        temp_patch_list.append(CutImage)
                        x_list.append(i)
                        y_list.append(j)
                        temp_x.append(i)
                        temp_y.append(j)
                        counter += 1
                        batchsize = 256

                        if counter == batchsize:

                            Dataset = SurvivalImageDataset(temp_patch_list, temp_x, temp_y, transform)
                            dataloader = torch.utils.data.DataLoader(Dataset, batch_size=batchsize, num_workers=0,
                                                                     drop_last=False)
                            for sample_img in dataloader:
                                images = sample_img['image']
                                images = images.to(device)
                                with torch.set_grad_enabled(False):
                                    preds_list, values_list ,out= Prediction(model_ft, images,device)
                                    # features = model_ft(images)

                            if inside_counter == 0:
                                #feature_list = np.concatenate(
                                 #   (features.cpu().detach().numpy(), classifier.cpu().detach().numpy()), axis=1)
                                feature_list = out
                                temp_x = np.reshape(np.array(temp_x), (len(temp_x), 1))
                                temp_y = np.reshape(np.array(temp_y), (len(temp_x), 1))

                                x_y_list = np.concatenate((temp_x, temp_y), axis=1)
                            else:
                                #feature_list = np.concatenate((feature_list, np.concatenate(
                                 #   (features.cpu().detach().numpy(), classifier.cpu().detach().numpy()), axis=1)),
                                  #                            axis=0)
                                feature_list = np.concatenate((feature_list, out),axis=0)
                                temp_x = np.reshape(np.array(temp_x), (len(temp_x), 1))
                                temp_y = np.reshape(np.array(temp_y), (len(temp_x), 1))

                                x_y_list = np.concatenate((x_y_list,
                                                           np.concatenate((temp_x, temp_y), axis=1)), axis=0)
                            inside_counter += 1
                            temp_patch_list = []
                            temp_x = []
                            temp_y = []
                            counter = 0

                        pbar_image.update()

            if counter < batchsize and counter > 0:
                Dataset = SurvivalImageDataset(temp_patch_list, temp_x, temp_y, transform)
                dataloader = torch.utils.data.DataLoader(Dataset, batch_size=batchsize, num_workers=0, drop_last=False)
                for sample_img in dataloader:
                    images = sample_img['image']
                    images = images.to(device)
                    with torch.set_grad_enabled(False):
                        # classifier, features = model_ft(images)
                        preds_list, values_list ,out= Prediction(model_ft, images,device)


                    #feature_list = np.concatenate((feature_list, np.concatenate(
                     #   (features.cpu().detach().numpy(), classifier.cpu().detach().numpy()), axis=1)), axis=0)
                    feature_list = np.concatenate((feature_list, out),axis=0)
                    temp_x = np.reshape(np.array(temp_x), (len(temp_x), 1))
                    temp_y = np.reshape(np.array(temp_y), (len(temp_x), 1))

                    x_y_list = np.concatenate((x_y_list,
                                               np.concatenate((temp_x, temp_y), axis=1)), axis=0)
                temp_patch_list = []
                temp_x = []
                temp_y = []
                counter = 0

        feature_df = pd.DataFrame.from_dict(feature_list)
        coordinate_df = pd.DataFrame({'X': x_y_list[:, 0], 'Y': x_y_list[:, 1]})
        graph_dataframe = pd.concat([coordinate_df, feature_df], axis=1)
        graph_dataframe = graph_dataframe.sort_values(by=['Y', 'X'])
        graph_dataframe = graph_dataframe.reset_index(drop=True)
        coordinate_df = graph_dataframe.iloc[:, 0:2]
        feature_df.to_csv(os.path.join(origin_dir, sample + '_feature_list.csv'))
        coordinate_df.to_csv(os.path.join(origin_dir, sample + '_node_location_list.csv'))
        index = list(graph_dataframe.index)
        graph_dataframe.insert(0, 'index_orig', index)

        node_dict = {}

        for i in range(len(coordinate_df)):
            node_dict.setdefault(i, [])

        X = max(set(np.squeeze(graph_dataframe.loc[:, ['X']].values, axis=1)))
        Y = max(set(np.squeeze(graph_dataframe.loc[:, ['Y']].values, axis=1)))
        del feature_df

        gridNum = 4
        X_size = int(X / gridNum)
        Y_size = int(Y / gridNum)

        with tqdm(total=(gridNum + 2) * (gridNum + 2)) as pbar:
            for p in range(gridNum + 2):
                for q in range(gridNum + 2):
                    if p == 0:
                        if q == 0:
                            is_X = graph_dataframe['X'] <= X_size * (p + 1)
                            is_X2 = graph_dataframe['X'] >= 0
                            is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                            is_Y2 = graph_dataframe['Y'] >= 0
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                        elif q == (gridNum + 1):
                            is_X = graph_dataframe['X'] <= X_size * (p + 1)
                            is_X2 = graph_dataframe['X'] >= 0
                            is_Y = graph_dataframe['Y'] <= Y
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) - 2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                        else:
                            is_X = graph_dataframe['X'] <= X_size * (p + 1)
                            is_X2 = graph_dataframe['X'] >= 0
                            is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) - 2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    elif p == (gridNum + 1):
                        if q == 0:
                            is_X = graph_dataframe['X'] <= X
                            is_X2 = graph_dataframe['X'] >= (X_size * (p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                            is_Y2 = graph_dataframe['Y'] >= 0
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        elif q == (gridNum + 1):
                            is_X = graph_dataframe['X'] <= X
                            is_X2 = graph_dataframe['X'] >= (X_size * (p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) - 2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        else:
                            is_X = graph_dataframe['X'] <= X
                            is_X2 = graph_dataframe['X'] >= (X_size * (p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) - 2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    else:
                        if q == 0:
                            is_X = graph_dataframe['X'] <= X_size * (p + 1)
                            is_X2 = graph_dataframe['X'] >= (X_size * (p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                            is_Y2 = graph_dataframe['Y'] >= 0
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        elif q == (gridNum + 1):
                            is_X = graph_dataframe['X'] <= X_size * (p + 1)
                            is_X2 = graph_dataframe['X'] >= (X_size * (p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) - 2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        else:
                            is_X = graph_dataframe['X'] <= X_size * (p + 1)
                            is_X2 = graph_dataframe['X'] >= (X_size * (p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y_size * (q + 1)
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) - 2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                    if len(X_10) == 0:
                        pbar.update()
                        continue

                    coordinate_dataframe = X_10.loc[:, ['X', 'Y']]
                    X_10 = X_10.reset_index(drop=True)
                    coordinate_list = coordinate_dataframe.values.tolist()
                    index_list = coordinate_dataframe.index.tolist()

                    feature_dataframe = X_10[X_10.columns.difference(['index_orig', 'X', 'Y'])]
                    feature_list = feature_dataframe.values.tolist()
                    coordinate_matrix = euclidean_distances(coordinate_list, coordinate_list)
                    coordinate_matrix = np.where(coordinate_matrix > 2.9, 0, 1)
                    cosine_matrix = cosine_similarity(feature_list, feature_list)

                    Adj_list = (coordinate_matrix == 1).astype(int) * (cosine_matrix >= threshold).astype(int)

                    for c, item in enumerate(Adj_list):
                        for node_index in np.array(index_list)[item.astype('bool')]:
                            if node_index == index_list[c]:
                                pass
                            else:
                                node_dict[index_list[c]].append(node_index)

                    pbar.update()

        a_file = open(os.path.join(origin_dir, sample + '_node_dict.pkl'), "wb")
        pickle.dump(node_dict, a_file)
        a_file.close()
        dict_len_list = []

        for i in range(0, len(node_dict)):
            dict_len_list.append(len(node_dict[i]))

        arglist_strict = np.argsort(np.array(dict_len_list))
        arglist_strict = arglist_strict[::-1]

        for arg_value in arglist_strict:
            if arg_value in node_dict.keys():
                for adj_item in node_dict[arg_value]:
                    if adj_item in node_dict.keys():
                        node_dict.pop(adj_item)
                        arglist_strict = np.delete(arglist_strict, np.argwhere(arglist_strict == adj_item))

        for key_value in node_dict.keys():
            node_dict[key_value] = list(set(node_dict[key_value]))

        supernode_coordinate_x_strict = []
        supernode_coordinate_y_strict = []
        supernode_feature_strict = []

        supernode_relate_value = [supernode_coordinate_x_strict,
                                  supernode_coordinate_y_strict,
                                  supernode_feature_strict]

        whole_feature = graph_dataframe[graph_dataframe.columns.difference(['index_orig', 'X', 'Y'])]

        with tqdm(total=len(node_dict.keys())) as pbar_node:
            for key_value in node_dict.keys():
                supernode_relate_value[0].append(graph_dataframe['X'][key_value])
                supernode_relate_value[1].append(graph_dataframe['Y'][key_value])
                if len(node_dict[key_value]) == 0:
                    select_feature = whole_feature.iloc[key_value]
                else:
                    select_feature = whole_feature.iloc[node_dict[key_value] + [key_value]]
                    select_feature = select_feature.mean()
                if len(supernode_relate_value[2]) == 0:
                    temp_select = np.array(select_feature)
                    supernode_relate_value[2] = np.reshape(temp_select, (1, 2048))
                else:
                    temp_select = np.array(select_feature)
                    supernode_relate_value[2] = np.concatenate(
                        (supernode_relate_value[2], np.reshape(temp_select, (1, 2048))), axis=0)
                pbar_node.update()

        coordinate_integrate = pd.DataFrame({'X': supernode_relate_value[0], 'Y': supernode_relate_value[1]})
        coordinate_matrix1 = euclidean_distances(coordinate_integrate, coordinate_integrate)
        coordinate_matrix1 = np.where(coordinate_matrix1 > spatial_threshold, 0, 1)

        fromlist = []
        tolist = []

        with tqdm(total=len(coordinate_matrix1)) as pbar_pytorch_geom:
            for i in range(len(coordinate_matrix1)):
                temp = coordinate_matrix1[i, :]
                selectindex = np.where(temp > 0)[0].tolist()
                for index in selectindex:
                    fromlist.append(int(i))
                    tolist.append(int(index))
                pbar_pytorch_geom.update()

        edge_index = torch.tensor([fromlist, tolist], dtype=torch.long)
        x = torch.tensor(supernode_relate_value[2], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)

        node_dict = pd.DataFrame.from_dict(node_dict, orient='index')
        node_dict.to_csv(os.path.join(superpatch_dir, sample + '_' + str(threshold) + '.csv'))
        torch.save(data, os.path.join(superpatch_dir, sample + '_' + str(threshold) + '_graph_torch.pt'))


def Parser_main():
    parser = argparse.ArgumentParser(description="TEA-graph superpatch generation")
    parser.add_argument("--database", default='TCGA', help="Use in the savedir", type=str)
    parser.add_argument("--cancertype", default='KIRC', help="cancer type", type=str)
    parser.add_argument("--graphdir", default="G:/MTGACN/MTGACN/Sample_data_for_demo/Graph_test/",
                        help="graph save dir", type=str)
    parser.add_argument("--imagedir", default="G:/MTGACN/MTGACN/Sample_data_for_demo/raw/",
                        help="svs file location", type=str)
    parser.add_argument("--weight_path", default=None, help="pretrained weight path", type=str)
    parser.add_argument("--imagesize", default=256, help="crop image size", type=int)
    parser.add_argument("--threshold", default=0.75, help="cosine similarity threshold", type=float)
    parser.add_argument("--spatial_threshold", default=5.5, help="spatial threshold", type=float)
    parser.add_argument("--gpu", default='0', help="gpu device number", type=str)
    return parser.parse_args()


def main():
    Argument = Parser_main()
    cancer_type = Argument.cancertype
    database = Argument.database
    image_dir = Argument.imagedir
    save_dir = Argument.graphdir
    gpu = Argument.gpu
    files = os.listdir(image_dir)

    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, database)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, cancer_type)
    final_files = [os.path.join(image_dir, file) for file in files]
    final_files.sort(key=lambda f: os.stat(f).st_size, reverse=False)

    device = torch.device(int(gpu) if torch.cuda.is_available() else "cpu")
    # model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 2)
    # if Argument.weight_path is not None:
    #    weight_path = Argument.weight_path
    #    load_weight = torch.load(weight_path, map_location = device)
    #    model_ft.load_state_dict(load_weight)

    model_ft = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model_ft.transform_input = True
    model_ft.AuxLogits.fc = nn.Linear(768, 3)
    model_ft.fc = nn.Linear(2048, 3)
    model_ft.aux_logits = False
    model_ft.load_state_dict(
    torch.load('G:\AAApathology\program\checkpoint\InceptionV3\level0\InceptionV3_HMN_29_Acc-0.8983.pth'))
    #model_ft = models.inception_v3(weights=('pretrained', Inception_V3_Weights.IMAGENET1K_V1))
    n_inputs = model_ft.fc.in_features
    classifier = nn.Sequential(
        #nn.Linear(n_inputs, 1794)
    )
    model_ft.fc = classifier

    model_ft = model_ft.to(device)
    model_ft.eval()

    with tqdm(total=len(final_files)) as pbar_tot:
        for image in final_files:
            supernode_generation(image, model_ft, device, Argument, save_dir)
            pbar_tot.update()

    false_graph_filtering(4.3)


if __name__ == "__main__":
    main()