import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GCNConv, global_max_pool, global_mean_pool
import torch.nn.functional as F
import sys
sys.path.append('..')

drug_num = 87
cline_num = 55


class HgnnEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HgnnEncoder, self).__init__()
        self.conv1 = HypergraphConv(in_channels, 128)
        self.batch1 = nn.BatchNorm1d(128)
        self.conv2 = HypergraphConv(128, 128)
        self.batch2 = nn.BatchNorm1d(128)
        self.conv3 = HypergraphConv(128, out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.drop_out = nn.Dropout(0.3)

        self.conv1_stru = HypergraphConv(in_channels, 128)
        self.batch1_stru = nn.BatchNorm1d(128)
        self.conv2_stru = HypergraphConv(128, 128)
        self.batch2_stru = nn.BatchNorm1d(128)
        self.conv3_stru = HypergraphConv(128, out_channels)


        self.conv1_chem = HypergraphConv(in_channels, 128)
        self.batch1_chem = nn.BatchNorm1d(128)
        self.conv2_chem = HypergraphConv(128, 128)
        self.batch2_chem = nn.BatchNorm1d(128)
        self.conv3_chem = HypergraphConv(128, out_channels)
    def forward(self, x,y,z, edge):

        x = self.act(self.conv1(x, edge))
        x = self.batch1(x)
        x = self.drop_out(x)
        x = self.act(self.conv2(x, edge))

        y = self.act(self.conv1_stru(y, edge))
        y = self.batch1_stru(y)
        y = self.drop_out(y)
        y = self.act(self.conv2_stru(y, edge))

        z = self.act(self.conv1_chem(z, edge))
        z = self.batch1_chem(z)
        z = self.drop_out(z)
        z = self.act(self.conv2_chem(z, edge))
        return x,y,z



class MultiModalAttentionFusion(nn.Module):
    def __init__(self,input_dim):
        super(MultiModalAttentionFusion, self).__init__()
        self.stru_linear = nn.Linear(input_dim, input_dim)
        self.target_linear = nn.Linear(input_dim, input_dim)
        self.chem_linear = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
    def forward(self, stru_features, target_features,chem_features):
        x = F.adaptive_avg_pool1d(stru_features,1)
        x = self.relu(self.stru_linear(x.permute(1,0)))
        x = x.sum(dim = 1)

        y = F.adaptive_avg_pool1d(target_features,1)
        y = self.relu(self.target_linear(y.permute(1,0)))
        y = y.sum(dim = 1)

        z = F.adaptive_avg_pool1d(chem_features,1)
        z = self.relu(self.chem_linear(z.permute(1,0)))
        z = z.sum(dim = 1)
        weight = x+y+z
        attention_weight = [x/weight,y/weight,z/weight]
        print(f'w1:{attention_weight[0]},w2:{attention_weight[1]},w3:{attention_weight[2]}')
        fused_features = torch.cat((attention_weight[0]* stru_features,attention_weight[1] * target_features,attention_weight[2] * chem_features),1)

        return fused_features

class BioEncoder(nn.Module):
    def __init__(self, dim_drug_stru,dim_drug_chem,dim_drug_target,dim_cellline, output, use_GMP=True):
        super(BioEncoder, self).__init__()
        #stru
        self.use_GMP = use_GMP
        self.conv1 = GCNConv(dim_drug_stru, 128)
        self.batch_conv1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, output)
        self.batch_conv2 = nn.BatchNorm1d(output)
        #chem
        self.fc_drug1_chem = nn.Linear(dim_drug_chem, 128)
        self.batch_drug1_chem = nn.BatchNorm1d(128)
        self.fc_drug2_chem = nn.Linear(128, output)
        #target
        self.fc_drug1_target = nn.Linear(dim_drug_target, 128)
        self.batch_drug1_target = nn.BatchNorm1d(128)
        self.fc_drug2_target = nn.Linear(128, output)

        self.drop_out = nn.Dropout(0.3)
        # -------cell line_layer
        self.fc_cell1 = nn.Linear(dim_cellline, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.fc_cell2 = nn.Linear(128, output)
        self.act = nn.ReLU()

    def forward(self, drug_stru_feature, drug_adj, ibatch,drug_chem_feature,drug_target_feature, gexpr_data):
        #stru
        x_drug_stru = self.conv1(drug_stru_feature, drug_adj)
        x_drug_stru = self.batch_conv1(self.act(x_drug_stru))
        x_drug_stru = self.drop_out(x_drug_stru)
        x_drug_stru = self.conv2(x_drug_stru, drug_adj)
        x_drug_stru = self.act(x_drug_stru)
        x_drug_stru = self.batch_conv2(x_drug_stru)
        if self.use_GMP:
            x_drug_stru = global_max_pool(x_drug_stru, ibatch)
        else:
            x_drug_stru = global_mean_pool(x_drug_stru, ibatch)
        # chem
        x_drug_chem = torch.tanh(self.fc_drug1_chem(drug_chem_feature))
        x_drug_chem = self.batch_drug1_chem(x_drug_chem)
        x_drug_chem = self.drop_out(x_drug_chem)
        x_drug_chem = self.act(self.fc_drug2_chem(x_drug_chem))
        # target
        x_drug_target = torch.tanh(self.fc_drug1_target(drug_target_feature))
        x_drug_target = self.batch_drug1_target(x_drug_target)
        x_drug_target = self.drop_out(x_drug_target)
        x_drug_target = self.act(self.fc_drug2_target(x_drug_target))
        # cellline
        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug_stru,x_drug_chem,x_drug_target,x_cellline


class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)
        self.batch = nn.BatchNorm1d(in_channels)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, merge_embed,druga_id, drugb_id, cellline_id):
        h = torch.cat((merge_embed[druga_id, :],merge_embed[drugb_id, :],merge_embed[cellline_id, :]), 1)
        h = self.act(self.fc1(h))
        h = self.batch1(h)
        h = self.drop_out(h)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        h = self.drop_out(h)
        h = self.fc3(h)
        return h.squeeze(dim=1)


class HypergraphSynergy(torch.nn.Module):
    def __init__(self, bio_encoder, graph_encoder,attention,decoder):
        super(HypergraphSynergy, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.attention = attention
        self.decoder = decoder
    def forward(self, drug_stru_feature, drug_adj, ibatch,drug_chem_feature,drug_target_feature,gexpr_data, adj, druga_id, drugb_id, cellline_id):
        drug_stru_embed,drug_chem_embed, drug_target_embed,cellline_embed = self.bio_encoder(drug_stru_feature, drug_adj, ibatch, drug_chem_feature,drug_target_feature, gexpr_data)
        chem_cell_embed = torch.cat((drug_chem_embed, cellline_embed), 0)
        target_cell_embed = torch.cat((drug_target_embed, cellline_embed), 0)
        stru_cell_embed = torch.cat((drug_stru_embed, cellline_embed), 0)
        graph_target_embed,graph_stru_embed , graph_chem_embed= self.graph_encoder(target_cell_embed,stru_cell_embed,chem_cell_embed, adj)
        fusion_embed = self.attention(graph_stru_embed,graph_target_embed,graph_chem_embed)
        res = self.decoder(fusion_embed,druga_id, drugb_id, cellline_id)
        return res