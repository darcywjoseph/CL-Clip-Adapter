import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def contrastive_loss(features, labels, temperature=0.07):

    features = F.normalize(features, dim=1)
    similarity = torch.matmul(features, features.T) / temperature

    labels = labels.unsqueeze(1)
    mask_positive = (labels == labels.T).float()
    mask_negative = (labels != labels.T).float()
    mask_positive.fill_diagonal_(0)

    exp_sim = torch.exp(similarity)
    positive_sum = (exp_sim * mask_positive).sum(dim=1)
    negative_sum = (exp_sim * mask_negative).sum(dim=1)

    eps = 1e-8
    invariant_loss = -torch.log((positive_sum + eps) / (positive_sum + negative_sum + eps))
    invariant_loss = invariant_loss.mean()

    return invariant_loss

def train_single_epoch(loader, 
                       model, 
                       optimizer, 
                       use_contrastive=True, 
                       w=0.5, device="cuda"):
    
    model.train()
    total_loss = 0
    total_samples = 0

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.to(device)

        features, logits = model(images)

        classif_loss = F.cross_entropy(logits, labels)

        if use_contrastive:
            contr_loss = contrastive_loss(features, labels)
            loss = classif_loss + w * contr_loss
        else:
            loss = classif_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples
    
