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

def get_domain_text_embs(model, cfg, source_text_prompts, target_text_prompts, class_names):
    """
    Gets the text embeddings of the prompts describing the source and target domains. 
    If generic is True, source_text_prompts and target_text_prompts are strings instead of 
    templates to put the class name in. 
    """
    if cfg.AUGMENTATION.GENERIC:
        text_embeddings = zeroshot_classifier(target_text_prompts, model, normalize=cfg.METHOD.NORMALIZE, model_type=cfg.EXP.IMAGE_FEATURES)
        text_embeddings = np.transpose(text_embeddings, (1,0))
        orig_prompts = text_embeddings
        if len(source_text_prompts) > 0:
            source_embeddings = zeroshot_classifier(source_text_prompts, model, normalize=cfg.METHOD.NORMALIZE, model_type=cfg.EXP.IMAGE_FEATURES)
            print("source emb before averaging", source_embeddings.shape)
            source_embeddings = source_embeddings.mean(dim=0)
            print("source emb after averaging", source_embeddings.shape)
            diffs = torch.stack([emb-source_embeddings[0] for emb in text_embeddings])
            diffs /= text_embeddings.norm(dim=-1, keepdim=True)
    else:
        print(target_text_prompts)
        # print("yo", len(source_text_prompts), len(source_text_prompts[0]))
        # go on a per class basis
        templates = target_text_prompts
        all_texts = []
        for t in source_text_prompts:
            texts = [[t.format(c)] for c in class_names]
            text_emb = zeroshot_classifier(texts, model, normalize=cfg.METHOD.NORMALIZE, model_type=cfg.EXP.IMAGE_FEATURES).T
            print(texts, "text_emb", text_emb.shape)
            all_texts.append(text_emb)
        if type(target_text_prompts[0]) == str:
            target_text_prompts = [target_text_prompts]
        print(target_text_prompts)
        for p in target_text_prompts:
            print(p)
            texts = [[t.format(c) for t in p] for c in class_names]
            text_emb = zeroshot_classifier(texts, model, normalize=cfg.METHOD.NORMALIZE, model_type=cfg.EXP.IMAGE_FEATURES).T
            all_texts.append(text_emb)
        # this subtracts the neutral embedding from the domain embeddings and normalizes. 
        text_pairs = torch.stack(all_texts)
        print("text pairs", text_pairs.shape)
        target_embeddings, source_embeddings = text_pairs, []
        if len(source_text_prompts) > 0:
            source_embeddings = text_pairs[:len(source_text_prompts)]
            target_embeddings = text_pairs[len(source_text_prompts):]
        else:
            source_embeddings = torch.zeros_like(target_embeddings)
        #     text_diffs = []
        #     source_domain = text_pairs[0]
        #     for target_domain in text_pairs[1:]:
        #         diff = target_domain - source_domain
        #         diff /= np.linalg.norm(diff, axis=-1, keepdims=True)
        #         # diff = np.expand_dims(diff, axis=0)
        #         text_diffs.append(diff)
        # else:
        #     target_embeddings = text_pairs
        #     text_diffs = text_pairs
        # diffs = torch.stack(text_diffs).permute(1,0,2) # should be (num_classes, num_domains, emb_size)
        # print("diffs shape", diffs.shape)
        # print("source embeddings", source_embeddings.shape)
        print("target embeddings", target_embeddings.shape)
    return source_embeddings, target_embeddings

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