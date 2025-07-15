import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix
import torch.nn.functional as f
import numpy as np
import os
from collections import Counter
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def g_loss(z, aug, temperature: float = 0.1, pos_only: bool = True):
    z = z.to(device)
    aug = aug.to(device)
    b = z.size(0)
    z = f.normalize(z, dim=-1)
    aug = f.normalize(aug, dim=-1)
    logits = torch.sum(z * aug, dim=1) / temperature
    logits = torch.mean(logits)
    logits_tensor = torch.tensor(logits.item()).unsqueeze(0)
    pos_mask = torch.zeros((b, b), dtype=torch.bool, device=device)
    pos_mask.fill_diagonal_(True)
    log_prob = logits_tensor if pos_only else logits_tensor - torch.log(logits_tensor.exp().sum(0, keepdim=True))
    pos_mask = pos_mask.to(device)
    log_prob = log_prob.to(device)
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)
    loss = -mean_log_prob_pos.mean()
    return loss


def calc_loss(x, x_aug, temperature=0.2, sym=True):
    # x and x_aug shape -> Batch x proj_hidden_dim
    x = x.clone().detach().requires_grad_(True).to(device)
    x_aug = x_aug.clone().detach().requires_grad_(True).to(device)
    x = x.view(x.size(0) // 116, 116, 116)
    x_aug = x_aug.view(x_aug.size(0) // 116, 116, 116)
    batch_size, _, _ = x.size()
    x = x.view(batch_size, -1)
    x_aug = x_aug.view(batch_size, -1)
    x_abs = x.norm(dim=1).to(device)
    x_aug_abs = x_aug.norm(dim=1).to(device)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)].unsqueeze(1)
    if sym:
        loss_0 = pos_sim / (sim_matrix.sum(dim=0, keepdim=True) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1, keepdim=True) - pos_sim)
        loss_0 = -torch.log(loss_0).mean()
        loss_1 = -torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_1 = - torch.log(loss_1).mean()
        return loss_1
    return loss


def loss_z(z, aug, temperature: float = 0.1, sym: bool = True):
    z = f.normalize(z, dim=-1)  
    aug = f.normalize(aug, dim=-1)
    batch_size = z.size(0)
    sim_matrix = torch.einsum('ik,jk->ij', z, aug)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)].unsqueeze(1)
    if sym:
        loss_0 = pos_sim / (sim_matrix.sum(dim=0, keepdim=True) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1, keepdim=True) - pos_sim)
        loss_0 = -torch.log(loss_0).mean()
        loss_1 = -torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss_1).mean()
    return loss


def loss_G(x, x_aug, temperature=0.2):
    x = x.clone().detach().requires_grad_(True).to(device)
    x_aug = x_aug.clone().detach().requires_grad_(True).to(device)
    x = x.view(x.size(0) // 116, 116, 116)
    x_aug = x_aug.view(x_aug.size(0) // 116, 116, 116)
    batch_size, _, _ = x.size()
    x = x.view(batch_size, -1)
    x_aug = x_aug.view(batch_size, -1)
    x_abs = x.norm(dim=1).to(device)
    x_aug_abs = x_aug.norm(dim=1).to(device)
    sim_matrix = torch.einsum('ik,ik->i', x, x_aug) / (x_abs * x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    loss = -torch.log(sim_matrix).mean()
    return loss


def train(model, train_loader, optimizer, criterion, lambda_):
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data
        labels = data.y
        optimizer.zero_grad()
        z1, z2, output, x, x_aug = model(data)
        loss1 = loss_z(z1, z2)
        loss2 = loss_G(x, x_aug)
        loss3 = criterion(output, labels)
        loss = lambda_ * (loss1 - loss2) + (1 - lambda_) * loss3
        total_loss += loss.item()
        assert isinstance(loss, torch.Tensor), f"Expected a tensor, but got {type(loss)}"
        loss.backward()
        optimizer.step()

    train_loss = total_loss / len(train_loader)
    return train_loss


def validate(model, test_loader, criterion, threshold):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []
    scores = []
    with torch.no_grad():
        for data in test_loader:
            data = data
            _, _, output, _, _ = model(data)
            for item in output:
                score = item[1].item()
                scores.append(score)

            positive_probs = output[:, 1]

            predicted = (positive_probs >= threshold).long()
            loss = criterion(output, data.y)
            total_loss += loss.item()

            correct += predicted.eq(data.y).sum().item()
            total += data.y.size(0)

            true_labels.extend(data.y.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)

    return total_loss / len(test_loader), accuracy,  true_labels, predicted_labels, scores


def majority_vote(predictions):

    transposed_predictions = list(zip(*predictions))
    final_scores = [np.mean(sample_preds) for sample_preds in transposed_predictions]
    final_predictions = [1 if score > 0.5 else 0 for score in final_scores]

    return final_predictions, final_scores


def calculate_accuracy(true_labels, predicted_labels):
    correct_predictions = sum(y__t == y_p for y__t, y_p in zip(true_labels, predicted_labels))
    accuracy = correct_predictions / len(true_labels)
    return accuracy


def compute_class_weights(data):
    labels = [data[i].y.cpu().numpy() for i in range(len(data))]
    class_counts = np.bincount(labels)
    total_samples = sum(class_counts)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = np.exp(class_weights)

    return class_weights




