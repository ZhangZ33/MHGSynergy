import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from model_reg  import BioEncoder, HypergraphSynergy, HgnnEncoder,Decoder,MultiModalAttentionFusion
from sklearn.model_selection import KFold
import os
import glob
import sys
sys.path.append('..')
from drug_util import GraphDataset, collate
from utils import metrics_graph_reg, set_seed_all
from process_data import getData

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Run on {device}!")

def load_data(dataset):
    cline_fea, drug_smiles_fea, drug_stru_fea,drug_chem_fea,drug_target_fea,gene_data, synergy = getData(dataset)
    cline_fea = torch.from_numpy(cline_fea).to(device)

    return  drug_stru_fea,drug_chem_fea,drug_target_fea,cline_fea, synergy



def data_split(synergy, rd_seed=0):
    synergy_pos = pd.DataFrame([i for i in synergy])
    train_size = 0.9
    synergy_cv_data, synergy_test = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),
                                             [int(train_size * len(synergy_pos))])
    np.random.shuffle(synergy_cv_data)
    np.random.shuffle(synergy_test)
    np.savetxt(path + 'test_y_true.txt', synergy_test[:, 3])
    test_label = torch.from_numpy(np.array(synergy_test[:, 3], dtype='float32')).to(device)
    test_ind = torch.from_numpy(synergy_test).type(torch.LongTensor).to(device)
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
    rmse, r2, pr = metrics_graph_reg(true_ls, pre_ls)
    return [rmse, r2, pr], loss_train


def test(drug_stru_set,drug_chem_set,drug_target_set,cline_fea_set, synergy_adj, index, label):
    model.eval()
    with torch.no_grad():
        for batch, (drug_stru,drug_chem,drug_target,cline) in enumerate(zip(drug_stru_set,drug_chem_set,drug_target_set,cline_fea_set)):
            pred = model(drug_stru.x, drug_stru.edge_index, drug_stru.batch,drug_chem[0],drug_target[0],cline[0], synergy_adj,
                                              index[:, 0], index[:, 1], index[:, 2])
        loss = loss_func(pred, label)
        rmse_test, r2_test, pr_test = metrics_graph_reg(label.cpu().detach().numpy(),
                                                              pred.cpu().detach().numpy())
        return [rmse_test, r2_test, pr_test], loss.item(), pred.cpu().detach().numpy()


if __name__ == '__main__':
    dataset_name = 'ALMANAC'            # or ALMANAC
    # dataset_name = 'ONEIL'
    seed = 0
    cv_mode_ls = [1, 2, 3]
    epochs = 4000
    learning_rate = 4e-3
    L2 = 1e-3
    alpha = 0.4
    for cv_mode in cv_mode_ls:
        path = 'result/' + dataset_name + '_' + str(cv_mode) + '_'
        file = open(path + 'result_d2d.txt', 'w')
        set_seed_all(seed)
        drug_stru_feature,drug_chem_feature,drug_target_feature,cline_feature, synergy_data = load_data(dataset_name)
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
        synergy_cv, index_test, label_test = data_split(synergy_data)
        if cv_mode == 1:
            cv_data = synergy_cv
        elif cv_mode == 2:
            cv_data = np.unique(synergy_cv[:, 2])
        else:
            cv_data = np.unique(np.vstack([synergy_cv[:, 0], synergy_cv[:, 1]]), axis=1).T
        # 5CV
        final_metric = np.zeros(3)
        fold_num = 0
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, validation_index in kf.split(cv_data):
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
            np.savetxt(path + 'val_' + str(fold_num) + '_true.txt', synergy_validation[:, 3])
            label_train = torch.from_numpy(np.array(synergy_train[:, 3], dtype='float32')).to(device)
            label_validation = torch.from_numpy(np.array(synergy_validation[:, 3], dtype='float32')).to(device)
            index_train = torch.from_numpy(synergy_train).type(torch.LongTensor).to(device)
            index_validation = torch.from_numpy(synergy_validation).type(torch.LongTensor).to(device)
            synergy_train_tmp = np.copy(synergy_train)
            edge_data = synergy_train_tmp[:, 0:3]
            synergy_edge = edge_data.reshape(1, -1)
            index_num = np.expand_dims(np.arange(len(edge_data)), axis=-1)
            synergy_num = np.concatenate((index_num, index_num, index_num), axis=1)
            synergy_num = np.array(synergy_num).reshape(1, -1)
            synergy_graph = np.concatenate((synergy_edge, synergy_num), axis=0)
            synergy_graph = torch.from_numpy(synergy_graph).type(torch.LongTensor).to(device)

            model = HypergraphSynergy(BioEncoder(dim_drug_stru=75, dim_drug_chem=drug_chem_feature.size(-1),
                                                 dim_drug_target=drug_target_feature.size(-1),
                                                 dim_cellline=cline_feature.shape[-1], output=128),
                                      HgnnEncoder(in_channels=128, out_channels=128),
                                      MultiModalAttentionFusion(input_dim=70), Decoder(in_channels=1152)).to(device)
            loss_func = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)

            best_metric = [0, 0, 0]
            best_epoch = 0
            for epoch in range(epochs):
                model.train()
                train_metric, train_loss = train(drug_stru_set,drug_chem_set,drug_target_set, cline_set, synergy_graph,
                                                 index_train, label_train)
                val_metric, val_loss, _ = test(drug_stru_set,drug_chem_set,drug_target_set,cline_set, synergy_graph,
                                               index_validation, label_validation)
                if epoch % 20 == 0:
                    print('Epoch: {:05d},'.format(epoch), 'loss_train: {:.6f},'.format(train_loss),
                          'RMSE: {:.6f},'.format(train_metric[0]), 'R2: {:.6f},'.format(train_metric[1]),
                          'Pearson r: {:.6f},'.format(train_metric[2]))

                    print('Epoch: {:05d},'.format(epoch), 'loss_val: {:.6f},'.format(val_loss),
                          'RMSE: {:.6f},'.format(val_metric[0]), 'R2: {:.6f},'.format(val_metric[1]),
                          'Pearson r: {:.6f},'.format(val_metric[2]))
                torch.save(model.state_dict(), '{}.pth'.format(epoch))
                if val_metric[2] > best_metric[2]:
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
                  'RMSE: {:.6f},'.format(best_metric[0]),
                  'R2: {:.6f},'.format(best_metric[1]), 'Pearson r: {:.6f},'.format(best_metric[2]))
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
        print('Final 5-cv average results, RMSE: {:.6f},'.format(final_metric[0]),
              'R2: {:.6f},'.format(final_metric[1]),
              'Pearson r: {:.6f},'.format(final_metric[2]))

