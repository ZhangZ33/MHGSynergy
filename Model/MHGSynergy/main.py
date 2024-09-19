import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from model  import BioEncoder, HypergraphSynergy, HgnnEncoder,Decoder,MultiModalAttentionFusion
from sklearn.model_selection import KFold
import os
import glob
import sys
sys.path.append('..')
from drug_util import GraphDataset, collate
from utils import metrics_graph, set_seed_all
from process_data import getData
import copy
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Run on {device}!")

def load_data(dataset):
    cline_fea, drug_smiles_fea, drug_stru_fea,drug_chem_fea,drug_target_fea,gene_data, synergy = getData(dataset)
    cline_fea = torch.from_numpy(cline_fea).to(device)
    threshold = 30

    for row in synergy:
        row[3] = 1 if row[3] >= threshold else 0


    return  drug_smiles_fea,drug_stru_fea,drug_chem_fea,drug_target_fea,cline_fea, synergy



def data_split(synergy, rd_seed=0):
    synergy_pos = pd.DataFrame([i for i in synergy if i[3] == 1])
    synergy_neg = pd.DataFrame([i for i in synergy if i[3] == 0])
    train_size = 0.9
    synergy_cv_pos, synergy_test_pos = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_pos))])
    synergy_cv_neg, synergy_test_neg = np.split(np.array(synergy_neg.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_neg))])
    synergy_cv_data = np.concatenate((np.array(synergy_cv_neg), np.array(synergy_cv_pos)), axis=0)
    synergy_test = np.concatenate((np.array(synergy_test_neg), np.array(synergy_test_pos)), axis=0)
    np.random.shuffle(synergy_cv_data)
    np.random.shuffle(synergy_test)
    np.savetxt(path + 'test_y_true.txt', synergy_test[:, 3])
    np.savetxt(path + 'test_index.txt', synergy_test)
    test_label = torch.from_numpy(np.array(synergy_test[:, 3], dtype='float32')).to(device)
    test_ind = torch.from_numpy(np.array(synergy_test,dtype='int64')).to(device)
    return synergy_cv_data, test_ind, test_label


def train(drug_stru_set,drug_chem_set,drug_target_set,cline_fea_set, synergy_adj, index, label):
    loss_train = 0
    true_ls, pre_ls = [], []
    optimizer.zero_grad()
    for batch, (drug_stru,drug_chem,drug_target,cline) in enumerate(zip(drug_stru_set,drug_chem_set,drug_target_set,cline_fea_set)):
        pred = model(drug_stru.x, drug_stru.edge_index, drug_stru.batch,drug_chem[0],drug_target[0],cline[0], synergy_adj,
                                          index[:, 0], index[:, 1], index[:, 2])
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        true_ls += label_train.cpu().detach().numpy().tolist()
        pre_ls += pred.cpu().detach().numpy().tolist()
    auc_train, aupr_train, f1_train, acc_train = metrics_graph(true_ls, pre_ls)
    return [auc_train, aupr_train, f1_train, acc_train], loss_train


def test(drug_stru_set,drug_chem_set,drug_target_set,cline_fea_set, synergy_adj, index, label):
    model.eval()
    with torch.no_grad():
        for batch, (drug_stru,drug_chem,drug_target,cline) in enumerate(zip(drug_stru_set,drug_chem_set,drug_target_set,cline_fea_set)):
            pred = model(drug_stru.x, drug_stru.edge_index, drug_stru.batch,drug_chem[0],drug_target[0],cline[0], synergy_adj,
                                             index[:, 0], index[:, 1], index[:, 2])
        loss = loss_func(pred, label)
        auc_test, aupr_test, f1_test, acc_test = metrics_graph(label.cpu().detach().numpy(),
                                                              pred.cpu().detach().numpy())
        return [auc_test, aupr_test, f1_test, acc_test], loss.item(), pred.cpu().detach().numpy()


if __name__ == '__main__':
    dataset_name = 'ALMANAC'            # or ALMANAC
    # dataset_name = 'ONEIL'
    seed = 0
    cv_mode_ls = [1, 2, 3]
    epochs = 4000
    learning_rate = 0.0001
    L2 = 1e-4
    alpha = 0.4
    for cv_mode in cv_mode_ls:
        path = 'result/' + dataset_name + '_' + str(cv_mode) + '_'
        file = open(path + 'result_d2d.txt', 'w')
        set_seed_all(seed)
        drug_fp_feature, drug_stru_feature,drug_chem_feature,drug_target_feature,cline_feature, synergy_data = load_data(dataset_name)
        drug_fp_feature = np.array(drug_fp_feature)
        drug_fp = torch.tensor(drug_fp_feature).to(device)
        drug_chem_feature = np.array(drug_chem_feature)
        drug_chem_feature = torch.tensor(drug_chem_feature).to(device)
        drug_target_feature = np.array(drug_target_feature)
        drug_target_feature = torch.tensor(drug_target_feature).to(device)
        drug_stru_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_stru_feature),
                                   collate_fn=collate, batch_size=len(drug_stru_feature), shuffle=False)
        cline_set = Data.DataLoader(dataset=Data.TensorDataset(cline_feature),
                                    batch_size=len(cline_feature), shuffle=False)
        drug_chem_set = Data.DataLoader(dataset=Data.TensorDataset(drug_chem_feature),batch_size=len(drug_chem_feature),shuffle=False)
        drug_target_set = Data.DataLoader(dataset=Data.TensorDataset(drug_target_feature),batch_size=len(drug_target_feature),shuffle=False)
        drug_fp_set = Data.DataLoader(dataset=Data.TensorDataset(drug_fp),
                                     batch_size=len(drug_fp), shuffle=False)
        # split synergy
        synergy_cv, index_test, label_test = data_split(synergy_data)
        if cv_mode == 1:
            cv_data = synergy_cv
        elif cv_mode == 2:
            cv_data = np.unique(synergy_cv[:, 2])
        else:
            cv_data = np.unique(np.vstack([synergy_cv[:, 0], synergy_cv[:, 1]]), axis=1).T
        # ---5CV
        final_metric = np.zeros(4)
        fold_num = 0
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, validation_index in kf.split(cv_data):
            # construct train_set and validation_set
            if cv_mode == 1:  # grobal_level
                synergy_train, synergy_validation = cv_data[train_index], cv_data[validation_index]
            elif cv_mode == 2:  # cell line_level
                train_name, test_name = cv_data[train_index], cv_data[validation_index]
                synergy_train = np.array([i for i in synergy_cv if i[2] in train_name])
                synergy_validation = np.array([i for i in synergy_cv if i[2] in test_name])
            else:  # drug combination_level
                pair_train, pair_validation = cv_data[train_index], cv_data[validation_index]
                synergy_train = np.array(
                    [j for i in pair_train for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
                synergy_validation = np.array(
                    [j for i in pair_validation for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
            # construct dataset
            np.savetxt(path + 'val_' + str(fold_num) + '_true.txt', synergy_validation[:, 3])
            label_train = torch.from_numpy(np.array(synergy_train[:, 3], dtype='float32')).to(device)
            label_validation = torch.from_numpy(np.array(synergy_validation[:, 3], dtype='float32')).to(device)
            index_train = torch.from_numpy(np.array(synergy_train, dtype='int64')).to(device)
            index_validation = torch.from_numpy(np.array(synergy_validation, dtype='int64')).to(device)
            # construct hypergraph
            edge_data = synergy_train[synergy_train[:, 3] == 1, 0:3]
            synergy_edge = edge_data.reshape(1, -1)
            index_num = np.expand_dims(np.arange(len(edge_data)), axis=-1)
            synergy_num = np.concatenate((index_num, index_num, index_num), axis=1)
            synergy_num = np.array(synergy_num).reshape(1, -1)
            synergy_graph = np.concatenate((synergy_edge, synergy_num), axis=0)
            synergy_graph = torch.from_numpy(synergy_graph).type(torch.LongTensor).to(device)

            # model_build
            model = HypergraphSynergy(BioEncoder(dim_drug_stru=75,dim_drug_chem=drug_chem_feature.size(-1),dim_drug_target=drug_target_feature.size(-1),dim_cellline=cline_feature.shape[-1], output=128),
                                      HgnnEncoder(in_channels=128, out_channels=128),MultiModalAttentionFusion(input_dim=142),Decoder(in_channels=1152)).to(device)
            loss_func = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)
            # ---run
            best_metric = [0, 0, 0, 0]
            best_epoch = 0
            for epoch in range(epochs):
                model.train()
                train_metric, train_loss = train(drug_stru_set,drug_chem_set,drug_target_set, cline_set, synergy_graph,
                                                 index_train, label_train)
                val_metric, val_loss, _ = test(drug_stru_set,drug_chem_set,drug_target_set,cline_set, synergy_graph,
                                               index_validation, label_validation)
                if epoch % 20 == 0:
                    print('Epoch: {:05d},'.format(epoch), 'loss_train: {:.6f},'.format(train_loss),
                          'AUC: {:.6f},'.format(train_metric[0]), 'AUPR: {:.6f},'.format(train_metric[1]),
                          'F1: {:.6f},'.format(train_metric[2]), 'ACC: {:.6f},'.format(train_metric[3]),
                          )
                    print('Epoch: {:05d},'.format(epoch), 'loss_val: {:.6f},'.format(val_loss),
                          'AUC: {:.6f},'.format(val_metric[0]), 'AUPR: {:.6f},'.format(val_metric[1]),
                          'F1: {:.6f},'.format(val_metric[2]), 'ACC: {:.6f},'.format(val_metric[3]))
                torch.save(model.state_dict(), '{}.pth'.format(epoch))
                if val_metric[0] > best_metric[0]:
                    best_metric = val_metric
                    best_epoch = epoch
                files = glob.glob('*.pth')
                for f in files:
                    epoch_nb = int(f.split('.')[0])
                    if epoch_nb < best_epoch:
                        os.remove(f)
            files = glob.glob('*.pth')
            for f in files:
                epoch_nb = int(f.split('.')[0])
                if epoch_nb > best_epoch:
                    os.remove(f)
            print('The best results on validation set, Epoch: {:05d},'.format(best_epoch),
                  'AUC: {:.6f},'.format(best_metric[0]),
                  'AUPR: {:.6f},'.format(best_metric[1]), 'F1: {:.6f},'.format(best_metric[2]),
                  'ACC: {:.6f},'.format(best_metric[3]))
            model.load_state_dict(torch.load('{}.pth'.format(best_epoch)))
            val_metric, _, y_val_pred = test(drug_stru_set ,drug_chem_set,drug_target_set,cline_set, synergy_graph, index_validation, label_validation)
            test_metric, _, y_test_pred = test(drug_stru_set ,drug_chem_set,drug_target_set,cline_set, synergy_graph, index_test, label_test)
            np.savetxt(path + 'val_' + str(fold_num) + '_pred.txt', y_val_pred)
            np.savetxt(path + 'test_' + str(fold_num) + '_pred.txt', y_test_pred)
            file.write('val_metric:')
            for item in val_metric:
                file.write(str(item) + '\t')
            file.write('\ntest_metric:')
            for item in test_metric:
                file.write(str(item) + '\t')
            file.write('\n')
            final_metric += test_metric
            fold_num = fold_num + 1
        final_metric /= 5
        file.write('\nfinal_metric')
        for item in final_metric:
            file.write(str(item) + '\t')
        print('Final 5-cv average results, AUC: {:.6f},'.format(final_metric[0]),
              'AUPR: {:.6f},'.format(final_metric[1]),
              'F1: {:.6f},'.format(final_metric[2]), 'ACC: {:.6f},'.format(final_metric[3]))

