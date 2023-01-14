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

    features = torch.cat(all_features).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    groups = torch.cat(all_groups).cpu().numpy()
    domains = torch.cat(all_domains).cpu().numpy()
    filenames = np.array(all_filenames)

    return features, labels, groups, domains, filenames

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

def load_embeddings(cache_file, dataset):
    """
    Loads the embeddings from a file
    """
    save_dict = torch.load(cache_file)
    train_features, train_labels, train_groups, train_domains, train_filenames = save_dict['train_features'], save_dict['train_labels'], save_dict['train_groups'], save_dict['train_domains'], save_dict['train_filenames']
    val_features, val_labels, val_groups, val_domains, val_filenames = save_dict['val_features'], save_dict['val_labels'], save_dict['val_groups'], save_dict['val_domains'], save_dict['val_filenames']
    test_features, test_labels, test_groups, test_domains, test_filenames = save_dict['test_features'], save_dict['test_labels'], save_dict['test_groups'], save_dict['test_domains'], save_dict['test_filenames']
    if dataset != 'ColoredMNISTBinary':
        old_val_features, old_val_labels, old_val_groups, old_val_domains, old_val_filenames = val_features, val_labels, val_groups, val_domains, val_filenames
        val_features, val_labels, val_groups, val_domains, val_filenames = val_features[::2], val_labels[::2], val_groups[::2], val_domains[::2], val_filenames[::2]
        test_features, test_labels, test_groups, test_domains, test_filenames = np.concatenate((test_features, old_val_features[1::2])), np.concatenate((test_labels, old_val_labels[1::2])), np.concatenate((test_groups, old_val_groups[1::2])), np.concatenate((test_domains, old_val_domains[1::2])), np.concatenate((test_filenames, old_val_filenames[1::2]))
    return train_features, train_labels, train_groups, train_domains, train_filenames, val_features, val_labels, val_groups, val_domains, val_filenames, test_features, test_labels, test_groups, test_domains, test_filenames

def projection(u, v):
    return (v * u).sum() / (u * u).sum() * u

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
    """ Computes CLIP text embeddings for a list of prompts. """
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


def get_ensamble_preds(val_features, probs, zeroshot_weights, dataset_domains=None):
    """
    Take in the clip image mebeddings, classfier, and clip ZS text embeddings, 
    averages the probabilites, and returns the predictions
    """
    print("ENSAMBLE PREDS", probs[0].shape)
    if dataset_domains is not None:
        soft_dom_label = np.matmul(val_features, dataset_domains.cpu().numpy())
        soft_dom_label = soft_dom_label.argmax(axis=1)

    # hacky: for MLP, model=output probs
    try:
        outputs = probs.cpu().numpy()
    except:
        outputs = probs
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