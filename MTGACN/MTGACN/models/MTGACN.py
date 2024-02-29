# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.PatchGCN import PatchGCN_module as gcn
from torch.nn import LayerNorm
from torch_geometric.nn import global_mean_pool, BatchNorm
from models.Modified_GAT import GATConv as GATConv
from torch_geometric.nn import GraphSizeNorm

from models.model_utils import weight_init
from models.model_utils import decide_loss_type
#from ceshi2 import MyViT
from models.pre_layer import preprocess
from models.post_layer import postprocess

class GAT_module(torch.nn.Module):

    def __init__(self, input_dim, output_dim, head_num, dropedge_rate, graph_dropout_rate, loss_type, with_edge, simple_distance, norm_type):
        """
        :param input_dim: Input dimension for GAT
        :param output_dim: Output dimension for GAT
        :param head_num: number of heads for GAT
        :param dropedge_rate: Attention-level dropout rate
        :param graph_dropout_rate: Node/Edge feature drop rate
        :param loss_type: Choose the loss type
        :param with_edge: Include the edge feature or not
        :param simple_distance: Simple multiplication of edge feature or not
        :param norm_type: Normalization method
        """

        super(GAT_module, self).__init__()
        self.conv = GATConv([input_dim, input_dim], output_dim, heads=head_num, dropout=dropedge_rate, with_edge=with_edge, simple_distance=simple_distance)
        self.norm_type = norm_type
        if norm_type == "layer":
            self.bn = LayerNorm(output_dim * int(self.conv.heads))
            self.gbn = None
        else:
            self.bn = BatchNorm(output_dim * int(self.conv.heads))
            self.gbn = GraphSizeNorm()
        self.prelu = decide_loss_type(loss_type, output_dim * int(self.conv.heads))
        self.dropout_rate = graph_dropout_rate
        self.with_edge = with_edge

    def reset_parameters(self):

        self.conv.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_attr, edge_index, batch):

        if self.training:
            drop_node_mask = x.new_full((x.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_node_mask = torch.bernoulli(drop_node_mask)
            drop_node_mask = torch.reshape(drop_node_mask, (1, drop_node_mask.shape[0]))
            drop_node_feature = x * drop_node_mask

            drop_edge_mask = edge_attr.new_full((edge_attr.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_edge_mask = torch.bernoulli(drop_edge_mask)
            drop_edge_mask = torch.reshape(drop_edge_mask, (1, drop_edge_mask.shape[0]))
            drop_edge_attr = edge_attr * drop_edge_mask
        else:
            drop_node_feature = x
            drop_edge_attr = edge_attr

        if self.with_edge == "Y":
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=drop_edge_attr, return_attention_weights=True)
        else:
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=None, return_attention_weights=True)
        out_x_temp = 0
        if self.norm_type == "layer":
            for c, item in enumerate(torch.unique(batch)):
                temp = self.bn(x_before[batch == item])
                if c == 0:
                    out_x_temp = temp
                else:
                    out_x_temp = torch.cat((out_x_temp, temp), 0)
        else:
            temp = self.gbn(self.bn(x_before), batch)
            out_x_temp = temp

        x_after = self.prelu(out_x_temp)

        return x_after, attention_value





class MTGACN(torch.nn.Module):

    def __init__(self, dropout_rate, dropedge_rate, Argument):
        super(MTGACN, self).__init__()
        torch.manual_seed(12345)
        self.Argument = Argument

        dim = Argument.initial_dim
        self.dropout_rate = dropout_rate
        self.dropedge_rate = dropedge_rate
        self.heads_num = Argument.attention_head_num
        self.include_edge_feature = Argument.with_distance
        self.layer_num = Argument.number_of_layers
        self.graph_dropout_rate = Argument.graph_dropout_rate
        self.residual = Argument.residual_connection
        self.norm_type = Argument.norm_type
        self.batch = Argument.batch_size

        self.gcn = gcn(Argument.initial_dim * Argument.attention_head_num, 1, Argument.dropout_rate)
        postNum = 0
        self.preprocess = preprocess(Argument)
        self.conv_list = nn.ModuleList([GAT_module(dim * self.heads_num, dim, self.heads_num, self.dropedge_rate,
                                                   self.graph_dropout_rate, Argument.loss_type,
                                                   with_edge=Argument.with_distance,
                                                   simple_distance=Argument.simple_distance,
                                                   norm_type=Argument.norm_type) for _ in
                                        range(int(Argument.number_of_layers))])
        postNum += int(self.heads_num) * len(self.conv_list)

        self.postprocess0 = postprocess(dim * self.heads_num, 0, dim * self.heads_num, (Argument.MLP_layernum - 1),dropout_rate)
        self.postprocess1 = postprocess(dim * self.heads_num, 0, dim * self.heads_num, (Argument.MLP_layernum-1), dropout_rate)
        self.postprocess2 = postprocess(dim * self.heads_num, 0, dim * self.heads_num,(Argument.MLP_layernum - 1), dropout_rate)

        self.encoder_layer2 = torch.nn.TransformerEncoderLayer(
            d_model=50,
            nhead=2,
            dropout=0.1,
            dim_feedforward=4 * 50,
        )
        self.encoder2 = torch.nn.TransformerEncoder(self.encoder_layer2, num_layers=8)

        self.decoder_layer2 = torch.nn.TransformerDecoderLayer(
            d_model=50,
            nhead=2,
            dropout=0.1,
            dim_feedforward=4 * 50,
        )
        self.decoder2 = torch.nn.TransformerDecoder(self.decoder_layer2, num_layers=8)

        self.encoder_layer1 = torch.nn.TransformerEncoderLayer(
            d_model=50,
            nhead=2,
            dropout=0.1,
            dim_feedforward=4 * 50,
        )
        self.encoder1 = torch.nn.TransformerEncoder(self.encoder_layer1, num_layers=8)

        self.decoder_layer1 = torch.nn.TransformerDecoderLayer(
            d_model=50,
            nhead=2,
            dropout=0.1,
            dim_feedforward=4 * 50,
        )
        self.decoder1 = torch.nn.TransformerDecoder(self.decoder_layer1, num_layers=8)

        self.encoder_layer0 = torch.nn.TransformerEncoderLayer(
            d_model=50,
            nhead=2,
            dropout=0.1,
            dim_feedforward=4 * 50,
        )
        self.encoder0 = torch.nn.TransformerEncoder(self.encoder_layer0, num_layers=8)

        self.decoder_layer0 = torch.nn.TransformerDecoderLayer(
            d_model=50,
            nhead=2,
            dropout=0.1,
            dim_feedforward=4 * 50,
        )
        self.decoder0 = torch.nn.TransformerDecoder(self.decoder_layer0, num_layers=8)

        self.conv1 = torch.nn.Conv1d(50,50,kernel_size=3,padding=0)
        self.risk_prediction_layer = nn.Linear(50, 1)



        #self.lstm=nn.LSTM(self.postprocess.postlayernum[-1],hidden_size=1,num_layers=1)
        #self.risk=nn.Linear(self.postprocess.postlayernum[-1],1)

    def reset_parameters(self):

        self.preprocess.reset_parameters()
        for i in range(int(self.Argument.number_of_layers)):
            self.conv_list[i].reset_parameters()
        self.postprocess.reset_parameters()
        self.lstm1.reset_parameters()
        self.lin1.reset_parameters()
        self.lstm2.reset_parameters()
        #self.risk_prediction_layer.reset_parameters()
        self.risk_prediction_layer.apply(weight_init)

    def forward(self, data, edge_mask=None, Interpretation_mode=False):
        #print('adj-t',data.adj_t.coo())
        #row, col, _ = data.adj_t.coo()
        #print(row.shape)
        preprocessed_input, preprocess_edge_attr = self.preprocess(data, edge_mask)
        #print('preprocessed_input',preprocessed_input.shape)
        #print('preprocess_edge_attr',preprocess_edge_attr.shape)
        batch = data.batch

        x0_glob = global_mean_pool(preprocessed_input, batch)
        x_concat0 = x0_glob
        x_concat1 = x0_glob
        x_concat2 = x0_glob
        x_out0 = preprocessed_input
        x_out1 = preprocessed_input
        x_out2 = preprocessed_input
        final_x0 = x_out0
        final_x1 = x_out1
        final_x2 = x_out2
        count = 0
        attention_list0 = []
        attention_list1 = []
        attention_list2 = []

        for i in range(1):
            select_idx = int(i)
            x_out_gcn0 = self.gcn(x_out0, data.adj_t)
            x_temp_out0, attention_value0 = \
                self.conv_list[select_idx](x_out0, preprocess_edge_attr, data.adj_t, batch)
            _, _, attention_value0 = attention_value0.coo()
            if len(attention_list0) == 0:
                attention_list0 = torch.reshape(attention_value0,
                                               (1, attention_value0.shape[0], attention_value0.shape[1]))
            else:
                attention_list0 = torch.cat((attention_list0, torch.reshape(attention_value0, (
                    1, attention_value0.shape[0], attention_value0.shape[1]))), 0)
            # print('x_temp',x_temp_out.shape)
            x_glob0 = global_mean_pool(x_temp_out0, batch)
            x_concat0 = torch.cat((x_concat0, x_glob0), 1)
            # print('x_glob',x_glob.shape)
            if self.residual == "Y":
                x_out0 = x_temp_out0 + x_out_gcn0
            else:
                x_out0 = x_temp_out0

            final_x1 = x_out1

        for i in range(2):
            select_idx = int(i)
            x_out_gcn1 = self.gcn(x_out1, data.adj_t)
            x_temp_out1, attention_value1 = \
                self.conv_list[select_idx](x_out1, preprocess_edge_attr, data.adj_t, batch)
            _, _, attention_value1 = attention_value1.coo()
            if len(attention_list1) == 0:
                attention_list1 = torch.reshape(attention_value1, (1, attention_value1.shape[0], attention_value1.shape[1]))
            else:
                attention_list1 = torch.cat((attention_list1, torch.reshape(attention_value1, (
                1, attention_value1.shape[0], attention_value1.shape[1]))), 0)
            #print('x_temp',x_temp_out.shape)
            x_glob1 = global_mean_pool(x_temp_out1, batch)
            x_concat1 = torch.cat((x_concat1, x_glob1), 1)
            #print('x_glob',x_glob.shape)
            if self.residual == "Y":
                x_out1 = x_temp_out1 + x_out_gcn1
            else:
                x_out1 = x_temp_out1

            final_x1 = x_out1


        for i in range(int(self.layer_num)):
            select_idx = int(i)
            x_out_gcn2 = self.gcn(x_out2, data.adj_t)
            x_temp_out2, attention_value2 = \
                self.conv_list[select_idx](x_out2, preprocess_edge_attr, data.adj_t, batch)
            _, _, attention_value2 = attention_value2.coo()
            if len(attention_list2) == 0:
                attention_list2 = torch.reshape(attention_value2, (1, attention_value2.shape[0], attention_value2.shape[1]))
            else:
                attention_list2 = torch.cat((attention_list2, torch.reshape(attention_value2, (
                1, attention_value2.shape[0], attention_value2.shape[1]))), 0)
            x_glob2 = global_mean_pool(x_temp_out2, batch)
            x_concat2 = torch.cat((x_concat2, x_glob2), 1)
            if self.residual == "Y":
                x_out2 = x_temp_out2 + x_out_gcn2
            else:
                x_out2 = x_temp_out2

            final_x2 = x_out2
            count = count + 1
        postprocessed_output0 = self.postprocess0(x_glob0, data.batch)
        postprocessed_output1 = self.postprocess1(x_glob1, data.batch)
        postprocessed_output2 = self.postprocess2(x_glob2, data.batch)
        postprocessed_output01 = self.encoder0(postprocessed_output0)
        postprocessed_output0 = self.decoder0(postprocessed_output0,postprocessed_output01)
        postprocessed_output11 = self.encoder1(postprocessed_output1)
        postprocessed_output1 = self.decoder1(postprocessed_output1,postprocessed_output11)
        postprocessed_output21 = self.encoder2(postprocessed_output2)
        postprocessed_output2 = self.decoder2(postprocessed_output2,postprocessed_output21)
        postprocessed_output0 = postprocessed_output0.resize(len(postprocessed_output0), 50, 1)
        postprocessed_output1 = postprocessed_output1.resize(len(postprocessed_output1), 50, 1)
        postprocessed_output2 = postprocessed_output2.resize(len(postprocessed_output2), 50, 1)
        postprocessed_output = torch.cat((postprocessed_output0, postprocessed_output1, postprocessed_output2), 2)
        postprocessed_output = self.conv1(postprocessed_output)
        postprocessed_output = postprocessed_output.resize(len(postprocessed_output0), 50)
        #postprocessed_output=torch.cat((postprocessed_output0,postprocessed_output1,postprocessed_output2),1)
        #postprocessed_output=postprocessed_output.resize(len(postprocessed_output),1,1,450)
        #postprocessed_output=self.conv1(postprocessed_output)
        #postprocessed_output=postprocessed_output.resize(len(postprocessed_output),450)
        risk = self.risk_prediction_layer(postprocessed_output)
        if Interpretation_mode:
            return risk, final_x2, attention_list2
        else:
            return risk