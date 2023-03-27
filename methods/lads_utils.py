from audioop import bias
import os
from tkinter import N
from xml import dom

from click import pass_obj
import clip
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.backends.cudnn as cudnn

import wandb

import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import pandas as pd
from transformers import AdamW
import omegaconf
from omegaconf import OmegaConf

from clip_utils import zeroshot_classifier
try:
    from progress_bar import progress_bar
except:
    progress_bar = lambda current, total, msg: None

import uuid


def get_domain_text_embs(model, cfg, source_text_prompts, target_text_prompts, class_names):
    """
    Gets the text embeddings of the prompts describing the source and target domains. 
    If generic is True, source_text_prompts and target_text_prompts are strings instead of 
    templates to put the class name in. 
    """
    print("len of prompts ", target_text_prompts, len(target_text_prompts), source_text_prompts, len(source_text_prompts))
    # if len(target_text_prompts) == 0 or len(source_text_prompts) == 0:
    #     return [], []
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
        templates = target_text_prompts
        all_texts = []
        for t in source_text_prompts:
            texts = [[t.format(c)] for c in class_names]
            text_emb = zeroshot_classifier(texts, model, normalize=cfg.METHOD.NORMALIZE, model_type=cfg.EXP.IMAGE_FEATURES).T
            all_texts.append(text_emb)
        if type(target_text_prompts[0]) == str:
            target_text_prompts = [target_text_prompts]
        for p in target_text_prompts:
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
            print("no source text prompts, using target text prompts for source embeddings")
            source_embeddings = torch.zeros_like(target_embeddings)
    return source_embeddings, target_embeddings


class EmbeddingDataset:
    """
    Takes in CLIP embeddings (INPUTS), labels, and CLIP text embedding (TEXT_EMB of shape (num_domains, clip emb shape)).
    Weakly labels the domain using the text embeddings 
    TODO: try softlabels
    """
    def __init__(self, cfg, inputs, labels, groups, dom_gt, text_emb=[]):
        self.inputs, self.labels, self.groups = inputs, labels, groups
        self.text_emb = text_emb
        self.cfg = cfg
        _, self.embedding_dim = inputs.shape
        if self.cfg.METHOD.USE_DOM_GT:
            print("==> Using domain GT labels")
            self.domain_labels = dom_gt
        else:
            print("==> Using domain CLIP labels")
            self.domain_labels = self.get_labels(self.text_emb, self.inputs) if len(self.text_emb) > 0 else np.array([0 for i in range(len(self.inputs))])
            print("domain labels", np.unique(np.array(self.domain_labels)))
        self.num_classes, self.num_domains = len(set(self.labels)), len(set(self.domain_labels))
        # get class weights for upweighting
        self.class_weights = self.get_counts(self.labels)
        self.dom_weights = self.get_counts(self.domain_labels)
        print(len(self.inputs), len(self.labels), len(self.domain_labels))
        assert len(self.inputs) == len(self.labels) == len(self.domain_labels), f"input, label, and domain label lengths don't match"

    @staticmethod
    def get_counts(labels):
        values, counts = np.unique(labels, return_counts=True)
        sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
        values, counts = [ list(tuple) for tuple in  sorted_tuples]
        fracs   = 1 / torch.Tensor(counts)
        return fracs / torch.max(fracs)

    @staticmethod
    def get_labels(text_emb, inputs):
        """ Gets weak domain labels given CLIP text embeddings """
        if len(text_emb.shape) == 3:
            text_emb = torch.mean(text_emb, dim = 0)
            print("new text emb ", text_emb.shape)
        similarity = (100.0 * torch.Tensor(inputs).to('cuda').float() @ text_emb.T.to('cuda').float()).softmax(dim=-1)
        values, indices = similarity.topk(1)
        return [i[0].item() for i in indices]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.groups[idx], self.domain_labels[idx]

class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)