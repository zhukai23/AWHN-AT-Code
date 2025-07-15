import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from load_data import BuildGraph
from model import TrainModel
from sklearn.model_selection import StratifiedKFold, ParameterGrid, KFold
from train import train, validate, calculate_accuracy, majority_vote, compute_class_weights  # data_choose
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, auc, accuracy_score
import random
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import ParameterGrid
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备:", device)
    seeds = []
    grid = {'lr': [], 'weight_decay': [], 'batch_size': [], 'epochs': [], 'randoms': seeds}
    param_grid = ParameterGrid(grid)
    n_neighbors =70
    build_graph = BuildGraph(n_neighbors)
    features, label = build_graph.load_data()
    num_nodes = 116
    linjie = build_graph.create_knn_graph(features, num_nodes)
    graph_data = []
    for i in range(len(features)):
        x = torch.as_tensor(features[i], dtype=torch.float).to(device)
        edge_index = torch.as_tensor(np.array(linjie[i].nonzero()), dtype=torch.long).to(device)
        y = torch.as_tensor(int(label[i]), dtype=torch.long).clone().detach().to(device)
        z = torch.as_tensor(linjie[i], dtype=torch.float).to(device)
        graph_data.append(Data(x=x, edge_index=edge_index, y=y, z=z))
    class_weights = compute_class_weights(graph_data)
    all_accuracies = []
    all_f1_scores = []
    all_aucs = []
    all_sens = []
    all_spes = []
    all_recalls = []
    all_y_t = []
    all_y_p = []
    all_scores = []
    for j in range(1):
        accuracies = []
        f1_scores = []
        aucs = []
        sens = []
        spes = []
        recalls = []
        y_ts = []
        y_ps = []
        scores = []
        Scos = []
        for params in param_grid:
            print(f"正在进行实验，参数: {params}")
            epochs = params['epochs']
            random_seed = params['randoms']
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            train_data, val_data = random_split(graph_data, [int(0.9 * len(graph_data)), len(graph_data) - int(0.9 * len(graph_data))],  # type: ignore
                                                generator=torch.Generator().manual_seed(random_seed))
            model = TrainModel().to(device)
            # criterion = nn.CrossEntropyLoss().to(device)
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights)).to(device)
            # def reset_parameters(model):
            #     model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            # reset_parameters(model)
            # optimizer = optim.SGD(model.parameters(), lr=params['lr'])
            optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            kf = KFold(n_splits=10, shuffle=True, random_state=32)
            best_fold_accuracy = 0
            best_fold_model_path = None
            pre_label = []
            y_t = []
            pre_score = []
            for fold, (train_index, test_index) in enumerate(kf.split(train_data)):
                train_fold = [train_data[i] for i in train_index]
                test_fold = [train_data[i] for i in test_index]
                train_loader = DataLoader(train_fold, batch_size=params['batch_size'], shuffle=False)
                test_loader = DataLoader(test_fold, batch_size=params['batch_size'], shuffle=False)
                best_epoch = 0
                best_accuracy = 0
                for epoch in range(epochs):
                    lambda_ = 1 - (epoch / epochs)
                    train_loss = train(model, train_loader, optimizer, criterion, lambda_)
                    val_loss, test_accuracy, _, _, _ = validate(model, test_loader, criterion, threshold=0.5)
                    if epoch == 0:
                        best_accuracy = test_accuracy
                        best_epoch = epoch
                    else:
                        if test_accuracy >= best_accuracy:
                            best_epoch = epoch
                            best_accuracy = test_accuracy
                            torch.save(model.state_dict(), fr"")
                model.load_state_dict(torch.load(fr""))
                _, accuracy_, y_t_, y_p_, score = validate(model, DataLoader(val_data, batch_size=params['batch_size'],
                                                                             shuffle=False), criterion, threshold=0.5)
                print("y_t:", y_t_, "\n\ny_p:", y_p_)
                pre_label.append(y_p_)
                pre_score.append(score)
                y_t.append(y_t_)
            y__t = y_t[0]
            y_p, Sco = majority_vote(pre_score, y__t)
            accuracy = calculate_accuracy(y__t, y_p)
            tn, fp, fn, tp = confusion_matrix(y__t, y_p).ravel()
            fpr, tpr, _ = roc_curve(y__t, Sco)
            Auc = auc(fpr, tpr)
            F1 = f1_score(y__t, y_p)
            Sen = tp / (tp + fn)
            Spe = tn / (tn + fp)
            print("y_t:", y__t, "\n\ny_p:", y_p)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1: {F1:.6f}")
            print(f"AUC: {Auc:.6f}")
            print(f"SEN: {Sen:.6f}")
            print(f"SPE: {Spe:.6f}")
            accuracies.append(accuracy)
            f1_scores.append(F1)
            aucs.append(Auc)
            sens.append(Sen)
            spes.append(Spe)
            y_ts.append(y__t)
            y_ps.append(y_p)
            Scos.append(Sco)
        all_accuracies.append(accuracies)
        all_f1_scores.append(f1_scores)
        all_aucs.append(aucs)
        all_sens.append(sens)
        all_spes.append(spes)
        all_y_t.append(y_ts)
        all_y_p.append(y_ps)
        all_scores.append(Scos)
        # results = {
        #     'y_true_': all_y_t,
        #     'scores_': all_scores
        # }
        # expanded_results = {
        #     'y_true' + str(i): sublist for i, sublist in enumerate(results['y_true_'])
        # }
        # expanded_results.update({
        #     'scores' + str(i): sublist for i, sublist in enumerate(results['scores_'])
        # })
        # results_df = pd.DataFrame(expanded_results)
        #
        # results_df.to_excel('', index=False)

    avg_accuracy = np.mean(all_accuracies)
    avg_f1_score = np.mean(all_f1_scores)
    avg_auc = np.mean(all_aucs)
    avg_sen = np.mean(all_sens)
    avg_spe = np.mean(all_spes)
    # std_accuracy = np.std(all_accuracies)
    # std_f1_score = np.std(all_f1_scores)
    # std_auc = np.std(all_aucs)
    # std_sen = np.std(all_sens)
    # std_spe = np.std(all_spes)
    print(f"总平均准确率: {avg_accuracy:.4f}")
    print(f"总平均f1分数: {avg_f1_score:.4f}")
    print(f"总平均auc: {avg_auc:.4f}")
    print(f"总平均sen: {avg_sen:.4f}")
    print(f"总平均spe: {avg_spe:.4f}")
