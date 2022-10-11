import torch
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as nnf
import torch.nn as nn
import torch.backends.cudnn as cudnn
from transformers import AdamW
import torchvision
# from helpers.clip_transformations import EmbeddingDataset
import wandb
from scipy.spatial import distance

import clip
import open_clip
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from scipy import stats
try:
    from progress_bar import progress_bar
except:
    progress_bar = lambda current, total, msg: None

import helpers.data_helpers as dh
import helpers

import omegaconf

def get_features(dataset, model, device, model_type):
    if model_type != 'clip' and model_type != 'openclip':
        return get_resnet_features(dataset, model, device)
    model.eval()
    all_features = []
    all_labels = []
    all_groups, all_domains = [], []
    all_filenames = []
    
    with torch.no_grad():
        for batch in tqdm(dataset):
            images, labels, groups, domains, filenames = batch['image'], batch['label'], batch['group'], batch['domain'], batch['filename']
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)
            all_groups.append(groups)
            all_domains.append(domains)
            all_filenames.extend(filenames)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy(), torch.cat(all_groups).cpu().numpy(), torch.cat(all_domains).cpu().numpy(), np.array(all_filenames)

def get_aug_features(dataset, model, device, text_prompts):
    all_features = []
    all_labels = []
    all_groups, all_domains = [], []
    print(text_prompts)
    
    
    with torch.no_grad():
        text_inputs = torch.cat([clip.tokenize(word[0]) for word in text_prompts]).to(device)
        # print(text_inputs)
        text_embeddings = model.encode_text(text_inputs)
        # get text embedding of "a photo of a digit" 
        print(len(text_inputs[0]))
        # print(type(text_prompts[0][0]))
        control = torch.cat([clip.tokenize("An image of a digit") ]).to(device)
        control_embedding = model.encode_text(control)
        for batch in tqdm(dataset):
            images, labels, groups, domains = batch['image'], batch['label'], batch['group'], batch['domain']
            features = model.encode_image(images.to(device))
            # append unaltered training data
            all_features.append(features)
            all_labels.append(labels)
            all_groups.append(groups) # group is the label 
            all_domains.append(domains) # red, green, blue, etc: label 
            # append training data combined with every text prompt embedding
            ref = features
            # print(len(text_embeddings[0]))
            # print(len(features[0]))
            for i in range(len(text_embeddings)):
                features = ref 
                # features = torch.add(features, text_embeddings[i])
                # features = torch.add(features, (features * text_embeddings[i] / text_embeddings[i]* text_embeddings[i]) * text_embeddings[i])
                features = torch.add(control_embedding, torch.add(features, text_embeddings[i])) / 3
                all_features.append(features)
                all_labels.append(labels)
                all_groups.append(groups)
                all_domains.append(domains)
        
    
    # print(torch.cat(all_features))
    first = torch.cat(all_features).cpu().numpy()
    
    return first, torch.cat(all_labels).cpu().numpy(), torch.cat(all_groups).cpu().numpy(), torch.cat(all_domains).cpu().numpy()

def get_resnet_features(dataset, model, device):
    """
    Gets the features of pretrained resnet model
    """
    features = []
    def hook(model, input, output):
        features.append(input[0].detach())
        return hook

    h = model.fc.register_forward_hook(hook)

    model.eval()
    all_features = []
    all_labels = []
    all_groups, all_domains = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataset):
            images, labels, groups, domains = batch['image'], batch['label'], batch['group'], batch['domain']
            out = model(images.to(device))
            all_labels.append(labels)
            all_groups.append(groups)
            all_domains.append(domains)
    all_features = features
    h.remove()
    print(torch.cat(all_features).cpu().numpy().shape)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy(), torch.cat(all_groups).cpu().numpy(), torch.cat(all_domains).cpu().numpy()

def projection(u, v):
    return (v * u).sum() / (u * u).sum() * u

def gram_schmidt(vv):
    """
    Creates orthonormal vectors which span the same space
    """
    nk = vv.size(1)
    uu = torch.zeros_like(vv, device=vv.device)
    print("shape ", uu.shape)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[:, k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu

def evaluate(predictions, labels, groups=[], label_names=None, num_augmentations=1):
    """
    Gets the evaluation metrics given the predictions and labels. 
    num_augmentations is for test-time augmentation, if its set >1, we group predictions by 
    num_augmentations and take the consesus as the label
    """
    if num_augmentations > 1:
        predictions_aug = predictions.reshape((int(len(predictions)/num_augmentations), num_augmentations))
        print("aug shape ", predictions_aug.shape, " label shape ", labels.shape)
        majority_pred = []
        for i, group in enumerate(predictions_aug):
            majority_pred.append(stats.mode(group)[0])
        predictions = np.array(majority_pred)
        print("new pred shape ", predictions.shape)

    cf_matrix = confusion_matrix(labels, predictions, labels=label_names)
    class_accuracy=100*cf_matrix.diagonal()/cf_matrix.sum(1)
    accuracy = np.mean((labels == predictions).astype(np.float)) * 100.
    balanced_acc = class_accuracy.mean()
    if len(groups) == 0:
        return accuracy, balanced_acc, np.array([round(c,2) for c in class_accuracy])
    else:
        group_acc = np.array([get_per_group_acc(value, predictions, labels, groups) for value in np.unique(groups)])
        return accuracy, balanced_acc, np.array([round(c,2) for c in class_accuracy]), np.array([round(g,2) for g in group_acc])

def get_per_group_acc(value, predictions, labels, groups):
    indices = np.array(np.where(groups == value))
    return np.mean((labels[indices] == predictions[indices]).astype(np.float)) * 100.

def zeroshot_classifier(prompts, model, normalize=True, model_type='clip', cuda_device='0'):
    model.eval()
    assert type(prompts[0]) == list, "prompts must be a list of lists"
    with torch.no_grad():
        zeroshot_weights = []
        for texts in tqdm(prompts):
            if model_type == 'clip':
                texts = clip.tokenize(texts).cuda() #tokenize
            elif model_type == 'openclip':
                texts = open_clip.tokenize(texts).cuda()
                texts = texts.to(torch.device(f'cuda:{cuda_device}'))
            # texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            if normalize:
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            if normalize:
                class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights.cpu()

def get_nn_metrics(aug_set, aug_domains, aug_labels, sample_set, sample_domains, sample_labels, num_samples=10):
    """
    Gets the nearest neighbors to calculate domain accuracy (how close the augmented data is to the domain),
    the class accuracy (how close the augmented data is to the desired class)
    """
    sample_set /= np.linalg.norm(sample_set, axis=-1, keepdims=True)
    sample_idxs = random.sample(list(range(len(aug_set))), num_samples)
    sim = aug_set @ sample_set.T
    neighbors = np.argmax(sim, axis=1)
    mean_cs = np.mean(np.max(sim, axis=1))
    prop_unique = len(np.unique(neighbors))/len(neighbors)
    neighbor_samples = [(x,y) for x,y in zip(sample_idxs, neighbors[sample_idxs])]
    neighbor_domains, neighbor_labels = [sample_domains[i] for i in neighbors], [sample_labels[i] for i in neighbors]
    domain_acc = np.mean((aug_domains == neighbor_domains).astype(np.float)) * 100.
    class_acc = np.mean((aug_labels == neighbor_labels).astype(np.float)) * 100.
    return neighbor_domains, neighbor_labels, domain_acc, class_acc, neighbor_samples, prop_unique, mean_cs

def get_clip_emb_drift(aug_set, aug_domains, sample_set, sample_domains, num_domains, val_set, val_domains):
    """
    For source domain S and target domain T,
    csim(I_S(test), I_T(test)) < csim(I_Aug)
    """
    # split test set
    # sample_set = sample_set / np.linalg.norm(sample_set, axis=-1, keepdims=True)
    # aug_set_norm = aug_set / np.linalg.norm(aug_set, axis=-1, keepdims=True)
    source_imgs = sample_set[np.where(sample_domains == 0)]
    source_imgs_norm = source_imgs / np.linalg.norm(source_imgs, axis=-1, keepdims=True)
    aug_set_norm = aug_set / np.linalg.norm(aug_set, axis=-1, keepdims=True)
    val_imgs_norm = val_set / np.linalg.norm(val_set, axis=-1, keepdims=True)
    metrics = {"cosine": {"clip": {"mean":[], "std": [], "max": [], "min": []}, "aug": {"mean":[], "std": [], "max": [], "min": []}}, "euclidean": {"clip": {"mean":[], "std": [], "max": [], "min": []}, "aug": {"mean":[], "std": [], "max": [], "min": []}}}
    
    similarities = distance.cdist(val_imgs_norm, source_imgs_norm, 'euclidean')
    nn_sim = np.min(similarities, axis=1)
    metrics['euclidean']['clip']['mean'].append(np.round(np.mean(nn_sim), 3))
    metrics['euclidean']['clip']['max'].append(np.round(np.max(nn_sim), 3))
    metrics['euclidean']['clip']['min'].append(np.round(np.min(nn_sim), 3))
    metrics['euclidean']['clip']['std'].append(np.round(np.std(nn_sim), 3))
    similarities = distance.cdist(aug_set_norm, source_imgs_norm, 'euclidean')
    nn_sim = np.min(similarities, axis=1)
    metrics['euclidean']['aug']['mean'].append(np.mean(nn_sim))
    metrics['euclidean']['aug']['max'].append(np.max(nn_sim))
    metrics['euclidean']['aug']['min'].append(np.min(nn_sim))
    metrics['euclidean']['aug']['std'].append(np.std(nn_sim))

    #cosine sim 
    nn_sim = np.max(val_imgs_norm @ source_imgs_norm.T, axis=1)  
    metrics['cosine']['clip']['mean'].append(np.round(np.mean(nn_sim), 3))
    metrics['cosine']['clip']['max'].append(np.round(np.max(nn_sim), 3))
    metrics['cosine']['clip']['min'].append(np.round(np.min(nn_sim), 3))
    metrics['cosine']['clip']['std'].append(np.round(np.std(nn_sim), 3))
    nn_sim = np.max(aug_set_norm @ source_imgs_norm.T, axis=1)  
    metrics['cosine']['aug']['mean'].append(np.round(np.mean(nn_sim), 3))
    metrics['cosine']['aug']['max'].append(np.round(np.max(nn_sim), 3))
    metrics['cosine']['aug']['min'].append(np.round(np.min(nn_sim), 3))
    metrics['cosine']['aug']['std'].append(np.round(np.std(nn_sim), 3))

    for domain in range(1, num_domains+1):
        target_imgs = sample_set[np.where(sample_domains == domain)]
        # cosine sim
        target_imgs_norm = target_imgs / np.linalg.norm(target_imgs, axis=-1, keepdims=True)
        similarities = distance.cdist(val_imgs_norm, target_imgs_norm, 'euclidean')
        nn_sim = np.min(similarities, axis=1)
        metrics['euclidean']['clip']['mean'].append(np.round(np.mean(nn_sim), 3))
        metrics['euclidean']['clip']['max'].append(np.round(np.max(nn_sim), 3))
        metrics['euclidean']['clip']['min'].append(np.round(np.min(nn_sim), 3))
        metrics['euclidean']['clip']['std'].append(np.round(np.std(nn_sim), 3))
        similarities = distance.cdist(aug_set_norm, target_imgs_norm, 'euclidean')
        nn_sim = np.min(similarities, axis=1)
        metrics['euclidean']['aug']['mean'].append(np.round(np.mean(nn_sim), 3))
        metrics['euclidean']['aug']['max'].append(np.round(np.max(nn_sim), 3))
        metrics['euclidean']['aug']['min'].append(np.round(np.min(nn_sim), 3))
        metrics['euclidean']['aug']['std'].append(np.round(np.std(nn_sim), 3))

        #cosine sim 
        nn_sim = np.max(val_imgs_norm @ target_imgs_norm.T, axis=1)  
        metrics['cosine']['clip']['mean'].append(np.round(np.mean(nn_sim), 3))
        metrics['cosine']['clip']['max'].append(np.round(np.max(nn_sim), 3))
        metrics['cosine']['clip']['min'].append(np.round(np.min(nn_sim), 3))
        metrics['cosine']['clip']['std'].append(np.round(np.std(nn_sim), 3))
        nn_sim = np.max(aug_set_norm @ target_imgs_norm.T, axis=1)  
        metrics['cosine']['aug']['mean'].append(np.round(np.mean(nn_sim), 3))
        metrics['cosine']['aug']['max'].append(np.round(np.max(nn_sim), 3))
        metrics['cosine']['aug']['min'].append(np.round(np.min(nn_sim), 3))
        metrics['cosine']['aug']['std'].append(np.round(np.std(nn_sim), 3))
        # breakpoint()

        print("cosine similarities ", nn_sim)
        
        return metrics


def get_ensamble_preds(val_features, model, zeroshot_weights, model_type='LR', dataset_domains=None):
    """
    Take in the clip image mebeddings, classfier, and clip ZS text embeddings, 
    averages the probabilites, and returns the predictions
    """
    if dataset_domains is not None:
        soft_dom_label = np.matmul(val_features, dataset_domains.cpu().numpy())
        soft_dom_label = soft_dom_label.argmax(axis=1)

    if model_type == 'LR':
        outputs = model.predict_proba(val_features)
    else:
        # hacky: for MLP, model=output probs
        try:
            outputs = model.cpu().numpy()
        except:
            outputs = model
    print(outputs.shape)
    salem_preds = np.argmax(outputs, axis=1)
    print(salem_preds.shape)
    # CLIP ZS
    zeroshot_weights = zeroshot_weights.cuda()
    images = torch.tensor(val_features).cuda()
    images /= images.norm(dim=-1, keepdim=True)
    # predict
    logits = (100. * images @ zeroshot_weights).float().softmax(dim=-1).cpu().numpy()
    print(logits.shape)
    zs_preds = np.argmax(logits, axis=1)
    print(zs_preds.shape)
    # average
    ensambled_outputs = np.mean([logits, outputs], axis=0)
    ensambled_preds = np.argmax(ensambled_outputs, axis=1)

    if dataset_domains is not None:
        dom_preds  = []
        for i in range(len(ensambled_preds)):
            if soft_dom_label[i] == 0:
                dom_preds.append(salem_preds[i])
            else:
                dom_preds.append(ensambled_preds[i])
        ret_preds = np.array(dom_preds)
    else:
        ret_preds = ensambled_preds

    return salem_preds, zs_preds, ensambled_preds, ret_preds

def get_pred_overlap(salem_preds, zs_preds, labels):
    """
    Get the overlap in correct predictions for salem and zeroshot.
    """
    salem_correct = np.where(salem_preds == labels)[0]
    zs_correct = np.where(zs_preds == labels)[0]
    print(len(salem_correct), len(zs_correct))
    print("salem correct ", salem_correct[:10])
    print("zs correct ", zs_correct[:10])
    salem_overlap = [i for i in salem_correct if i in zs_correct]
    salem_nonverlap = [i for i in salem_correct if not (i in zs_correct)]
    zs_nonverlap = [i for i in zs_correct if not (i in salem_correct)]
    num_zs_correct_nonoverlap = len(zs_correct) - len(salem_overlap)
    num_salem_correct_nonverlap = len(salem_correct) - len(salem_overlap)
    return num_salem_correct_nonverlap, num_salem_correct_nonverlap/len(labels), num_salem_correct_nonverlap/len(salem_correct)