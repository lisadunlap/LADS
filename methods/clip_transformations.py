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
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Tuple, Optional, Union
from sklearn.metrics import confusion_matrix
import omegaconf
from sklearn.decomposition import PCA
from pytorch_revgrad import RevGrad
try:
    from progress_bar import progress_bar
except:
    progress_bar = lambda current, total, msg: None

from snorkel.classification import cross_entropy_with_probs
import uuid

from methods.predictors import EmbeddingDebiasModel, MLP, Predictor
from clip_utils import zeroshot_classifier, projection, gram_schmidt, evaluate

from omegaconf import OmegaConf

from helpers.text_templates import imagenet_templates, part_templates, imagenet_templates_small
from helpers.data_helpers import DATASET_CLASSES, DATASET_DOMAINS
import helpers

device = "cuda" if torch.cuda.is_available() else "cpu"

class Noop:
    """
    Does nothing.

    Computes CLIP embeddings. If NEUTRAL_PROMPTS is not empty, then we take the difference between our prompts and the neutral prompt.
    If EXP.GENERAL=False, then we compute the embeddings for each class (num_text_prompts, )
    """
    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        if cfg.AUGMENTATION.GENERIC:
            print("using generic prompts")
            assert type(text_prompts[0]) == list
        else:
            assert type(text_prompts[0]) == str
        self.class_names = DATASET_CLASSES[cfg.DATA.DATASET]
        self.domain_names = DATASET_DOMAINS[cfg.DATA.DATASET]
        self.text_prompts = text_prompts
        self.cfg = cfg
        self.model = model
        self.neutral_prompts = neutral_prompts
        print(text_prompts, neutral_prompts)
        if self.cfg.AUGMENTATION.GENERIC:
            self.text_embeddings = zeroshot_classifier(text_prompts, model, normalize=self.cfg.METHOD.NORMALIZE, model_type=self.cfg.EXP.IMAGE_FEATURES).cpu().numpy()
            self.text_embeddings = np.transpose(self.text_embeddings, (1,0))
            self.orig_prompts = self.text_embeddings
            if len(self.neutral_prompts) > 0:
                self.neutral_embeddings = zeroshot_classifier(neutral_prompts, model, normalize=self.cfg.METHOD.NORMALIZE, model_type=self.cfg.EXP.IMAGE_FEATURES).cpu().numpy()
                self.text_embeddings = np.array([emb-self.neutral_embeddings[0] for emb in self.text_embeddings])
                self.text_embeddings = self.normalize(self.text_embeddings)
        else:
            # go on a per class basis
            templates = self.neutral_prompts + self.text_prompts
            all_texts = []
            for t in templates:
                texts = [[t.format(c)] for c in self.class_names]
                text_emb = self.normalize(zeroshot_classifier(texts, model, normalize=self.cfg.METHOD.NORMALIZE, model_type=self.cfg.EXP.IMAGE_FEATURES).cpu().numpy().T)
                all_texts.append(text_emb)
            # this subtracts the neutral embedding from the domain embeddings and normalizes. 
            text_pairs = np.array(all_texts)
            if len(self.neutral_prompts) > 0:
                self.neutral_embeddings = text_pairs[0]
                self.orig_prompts = text_pairs[1:]
                text_diffs = []
                source_domain = text_pairs[0]
                for target_domain in text_pairs[1:]:
                    diff = target_domain - source_domain
                    diff /= np.linalg.norm(diff, axis=-1, keepdims=True)
                    # diff = np.expand_dims(diff, axis=0)
                    text_diffs.append(diff)
            else:
                self.orig_prompts = text_pairs
                text_diffs = text_pairs
            print("text diff shape ", np.array(text_diffs).shape) # should be (num_domains, num_classes, emb_size)
            self.text_embeddings = np.array(text_diffs).transpose((1, 0, 2))
            print("text diff shape ", self.text_embeddings.shape)

    @staticmethod
    def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]

    @staticmethod
    def get_embedding(text_prompts, model):
        text_inputs = torch.cat([clip.tokenize(t) for t in text_prompts]).cuda()
        # Calculate features
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        return text_features.cpu().numpy()
    
    @staticmethod
    def normalize(inputs):
        # print("NORMALIZE BERFOR ", inputs.shape, np.linalg.norm(inputs, axis=-1, keepdims=True))
        try:
            inputs /= np.linalg.norm(inputs, axis=-1, keepdims=True)
            # inputs /= np.linalg.norm(inputs, axis=0)
        except:
            print("NOMRALIZE ERROR ", inputs, np.linalg.norm(inputs, axis=-1, keepdims=True))
        # print("NORMALIZE AFTER ", inputs.shape)
        return inputs

    def apply(self, inputs, labels=None):
        if self.cfg.METHOD.NORMALIZE:
            return self.normalize(inputs)
        return inputs

    def calc_dist(self, inputs, labels):
        assert len(self.text_prompts) > 0, "text prompts needed"
        ret = [["label", "text", "text id", "sim"]]
        for i in range(inputs.shape[0]):
            for j in range(len(self.text_prompts)):
                if self.cfg.AUGMENTATION.GENERIC:
                    dist = 1 - distance.cosine(inputs[i], self.text_embeddings[j])
                    ret.append([int(labels[i]), self.text_prompts[j][0], j, dist])
                else:
                    # average across classes
                    dist = 1 - distance.cosine(inputs[i], np.average(self.text_embeddings, axis=0)[j])
                    ret.append([int(labels[i]), self.text_prompts[j], j, dist])
        return pd.DataFrame(ret[1:], columns=ret[0])

class SubtractEmbeddings(Noop):
    """
    Simply subtracts the embeddings of the text prompts (averages the differences in concepts and image embeddings)
    """

    # def apply(self, inputs):
    #     inputs = super().apply(inputs)
    #     ret = []
    #     for i in range(inputs.shape[0]):
    #         altered = np.average(np.array([i - t for t in self.text_embeddings]), axis=0)
    #         ret.append(altered)
    #     return np.array(ret)

    def apply(self, inputs, labels=None):
        inputs = super().apply(inputs, labels)
        ret = []
        for h in inputs:
            if not self.cfg.AUGMENTATION.GENERIC:
                bias_cls = np.argmax((torch.Tensor(h).half() @ self.text_embeddings[:,0].T).float().numpy())
                b = self.text_embeddings[:,0][bias_cls]
                b /= np.linalg.norm(b, axis=-1, keepdim=True)
                proj = (1- self.cfg.METHOD.ALPHA) * h - self.cfg.METHOD.ALPHA * b
                proj /= np.linalg.norm(proj, axis=-1)
                ret.append(proj)
            else:
                altered = np.average(np.array([(h - t)/np.linalg.norm(h - t, axis=-1) for t in self.text_embeddings]), axis=0)
                ret.append(altered)
        return np.array(ret)

class CustomProjection(Noop):
    """
    Loads a dict of bias vectors (where the keys are the text prompts) and 
    applies vector projection to the relevant inputs (aka if we get an input
    with the highest cosine similarity to "green", we project out of the "green" vector)
    """
    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        # WARNING: prompts should be a list of lists
        if cfg.AUGMENTATION.GENERIC:
            assert type(text_prompts[0]) == list
        else:
            assert type(text_prompts[0]) == str
        print("TEXT PROMPTS ", text_prompts)
        self.cfg = cfg
        assert 'BIAS_VEC_FILE' in cfg.METHOD, "need a bias vector file"
        self.text_prompts = text_prompts
        self.model = model
        self.bias_dict = torch.load(self.cfg.METHOD.BIAS_VEC_FILE)
        self.bias_vectors = []
        print(text_prompts[:-1], list(self.bias_dict.keys()))
        for prompt in text_prompts[:-1]:
            flag = False
            for p in prompt:
                if p in self.bias_dict:
                    print("found! ", p)
                    self.bias_vectors.append(self.bias_dict[p])
                    flag = True
            if not flag:
                raise ValueError("prompt does not appear in keys")
        print(np.array(self.bias_vectors).shape)
        # create a vec of zeros for the last class, which is neutral
        self.bias_vectors.append(torch.zeros_like(self.bias_vectors[0]))
        self.text_embeddings = zeroshot_classifier(text_prompts, model, model_type=self.cfg.EXP.IMAGE_FEATURES).T
        b = gram_schmidt(self.text_embeddings.T)
        self.text_embeddings = b.T
        print("TEXT EMB SHAPE ", self.text_embeddings.shape)

    def apply(self, inputs, labels=None):
        inputs = super().apply(inputs, labels)
        ret = []
        for h in inputs:
            bias_cls = np.argmax((torch.Tensor(h).half() @ self.text_embeddings.T).float().numpy())
            b = self.bias_vectors[bias_cls].numpy()
            h_v = np.nan_to_num(projection(b, h))
            proj = h - h_v
            proj /= np.linalg.norm(proj, axis=-1)
            ret.append(h - h_v)
        return np.array(ret)

    def calc_dist(self, inputs, labels):
        assert len(self.text_prompts) > 0, "text prompts needed"
        ret = [["label", "text", "text id", "sim"]]
        if self.cfg.METHOD.NORMALIZE:
            inputs = self.normalize(inputs)
        for i in range(inputs.shape[0]):
            for j in range(len(self.text_prompts)):
                dist = 1 - distance.cosine(inputs[i], self.text_embeddings[j])
                ret.append([int(labels[i]), self.text_prompts[j][0], j, dist])
        return pd.DataFrame(ret[1:], columns=ret[0])

class PCAProjection(Noop):
    """
    Loads a dict of bias vectors (where the keys are the text prompts) and 
    applies vector projection to the relevant inputs (aka if we get an input
    with the highest cosine similarity to "green", we project out of the "green" vector)
    """
    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        super().__init__(text_prompts, model, cfg, neutral_prompts)
        pca = PCA(n_components=cfg.METHOD.K, svd_solver='full')
        pca.fit(self.text_embeddings)
        self.bias_space = pca.components_

    def apply(self, inputs, labels=None):
        inputs = super().apply(inputs, labels)
        ret = []
        for h in inputs:
            h_v = np.sum([(np.dot(h, b) / np.dot(b, b)) * b for b in self.bias_space], axis=0)
            norm_vec  = h - h_v
            ret.append(self.normalize(norm_vec))
        return np.array(ret)

class AvgProjection(Noop):
    """
    Loads a dict of bias vectors (where the keys are the text prompts) and 
    applies vector projection to the relevant inputs (aka if we get an input
    with the highest cosine similarity to "green", we project out of the "green" vector)
    """
    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        super().__init__(text_prompts, model, cfg, neutral_prompts)

    def apply(self, inputs, labels=[]):
        inputs = super().apply(inputs, labels)
        ret = []
        if len(labels) == 0:
            for h in inputs:
                h_v = np.sum([(np.dot(h, b) / np.dot(b, b)) * b for b in np.average(self.text_embeddings, axis=0)], axis=0)
                norm_vec  = (1-self.cfg.METHOD.ALPHA) * h - self.cfg.METHOD.ALPHA * h_v
                ret.append(self.normalize(norm_vec))
            return np.array(ret)
        else:
            for h, l in zip(inputs, labels):
                h_v = np.sum([(np.dot(h, b) / np.dot(b, b)) * b for b in self.text_embeddings[int(l)]], axis=0)
                norm_vec  = (1-self.cfg.METHOD.ALPHA) * h - self.cfg.METHOD.ALPHA * h_v
                ret.append(self.normalize(norm_vec))
            return np.array(ret)

class SentenceDebias(Noop):

    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        # WARNING: prompts should be a list of lists
        assert type(text_prompts[0]) == list
        self.prompts = text_prompts
        self.model = model
        self.d = len(self.prompts[0])
        self.embs = []
        self.bias_vectors = []
        self.text_embeddings = zeroshot_classifier(text_prompts, model, model_type=cfg.EXP.IMAGE_FEATURES).T
        print("TEXT EMB SHAPE ", self.text_embeddings.shape)
        self.cfg = cfg
        if self.cfg.METHOD.NORMALIZE:
            self.text_embeddings = self.normalize(self.text_embeddings)
        pca = PCA(n_components=cfg.METHOD.K, svd_solver='full')
        pca.fit(self.text_embeddings)
        self.bias_space = pca.components_
        print("BIASED EMB SHAPE ", self.bias_space.shape)

    def apply(self, inputs, labels=None):
        inputs = super().apply(inputs, labels)
        ret = []
        for h in inputs:
            h_v = np.sum([(np.dot(h, b) / np.dot(b, b)) * b for b in self.bias_space], axis=0)
            ret.append((h - h_v) / np.linalg.norm(h-h_v))
        return np.array(ret)

    def calc_dist(self, inputs, labels):
        assert len(self.prompts) > 0, "text prompts needed"
        ret = [["label", "text", "text id", "sim"]]
        if self.cfg.METHOD.NORMALIZE:
            inputs = self.normalize(inputs)
        for i in range(inputs.shape[0]):
            for j in range(len(self.prompts)):
                dist = 1 - distance.cosine(inputs[i], self.text_embeddings[j])
                ret.append([int(labels[i]), self.prompts[j][0], j, dist])
        return pd.DataFrame(ret[1:], columns=ret[0])

class HardDebias(Noop):

    def __init__(self, text_prompts, model, cfg, neutral_prompts):
        # WARNING: prompts should be a list of lists
        assert type(text_prompts[0]) == list
        self.text_prompts = text_prompts
        self.model = model
        self.d = len(self.text_prompts[0])
        self.embs = []
        self.cfg = cfg
        self.bias_vectors = []
        self.text_embeddings = zeroshot_classifier(text_prompts, model, model_type=self.cfg.EXP.IMAGE_FEATURES).cpu().numpy()
        self.text_embeddings = np.transpose(self.text_embeddings, (1,0))
        biased_sets = self.create_biased_subspace(text_prompts, model, emb_len=self.text_embeddings.shape[-1])
        pca, diffs = self.doPCA(biased_sets, cfg.METHOD.K)
        self.text_embeddings = diffs
        print("text emb shapes", self.text_embeddings.shape)
        self.bias_space = pca.components_
        print("BIASED SUBSPACE SHAPE ", self.bias_space.shape, self.text_embeddings.shape[-1])

    @staticmethod
    def create_biased_subspace(text_prompts, model, emb_len=512):
        text_emb = np.zeros((len(text_prompts), len(text_prompts[0]), emb_len))
        with torch.no_grad():
            for i in range(len(text_prompts)):
                for j in range(len(text_prompts[i])):
                    texts = clip.tokenize(text_prompts[i][j]).cuda() #tokenize
                    class_embeddings = model.encode_text(texts) #embed with text encoder
                    class_embeddings /= class_embeddings.norm()
                    text_emb[i][j] = torch.squeeze(class_embeddings).cpu().numpy()

        print("orignal shape", text_emb.shape)
        # biased_sets = np.transpose(text_emb, (1, 0, 2))
        return text_emb

    @staticmethod
    def doPCA(biased_sets, num_components = 10):
        centers = []
        for pairs in biased_sets:
            centers.append(pairs[1]-pairs[0])
        matrix = np.array(centers)
        pca = PCA(n_components = num_components)
        pca.fit(matrix)
        return pca, matrix

    @staticmethod
    def project(a, b):
        return (np.dot(a, b) / np.dot(b, b))*b

    def apply(self, inputs, labels=None):
        inputs = super().apply(inputs, labels)
        ret = []
        for h in inputs:
            h_v = np.sum([self.project(h, b) for b in self.bias_space], axis=0)
            orth_h = np.expand_dims(h - h_v, axis=0)
            orth_h /= np.linalg.norm(orth_h, axis=-1, keepdims=True)
            ret.append(orth_h[0])
        return np.array(ret)

    def calc_dist(self, inputs, labels):
        assert len(self.text_prompts) > 0, "text prompts needed"
        ret = [["label", "text", "text id", "sim"]]
        print(self.text_embeddings.shape)
        if self.cfg.METHOD.NORMALIZE:
            inputs = self.normalize(inputs)
        for i in range(inputs.shape[0]):
            for j in range(len(self.text_prompts)):
                dist = 1 - distance.cosine(inputs[i], self.text_embeddings[j])
                ret.append([int(labels[i]), self.text_prompts[j][-1], j, dist])
        return pd.DataFrame(ret[1:], columns=ret[0])
        
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
        self.num_classes, self.num_domains = len(set(self.labels)), len(set(self.domain_labels))
        # get class weights for upweighting
        self.class_weights = self.get_counts(self.labels)
        self.dom_weights = self.get_counts(self.domain_labels)
        assert len(self.inputs) == len(self.labels) == len(self.domain_labels), "input, label, and domain label lengths don't match"

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
        similarity = (100.0 * torch.Tensor(inputs).to(device).float() @ text_emb.T.to(device).float()).softmax(dim=-1)
        values, indices = similarity.topk(1)
        return [i[0].item() for i in indices]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # if self.cfg.METHOD.MODEL.WEAK_LABELS:
        #     soft_label = [1/self.num_domains for i in range(self.num_domains)]
        #     return self.inputs[idx], self.labels[idx], self.groups[idx], torch.Tensor(soft_label)
        return self.inputs[idx], self.labels[idx], self.groups[idx], self.domain_labels[idx]

class EmbeddingTextDataset(EmbeddingDataset):
    """
    Same as EmbeddingDataset but instead of returning the domain label, it returns the text embedding for the domain label.
    """
    def __init__(self, cfg, inputs, labels, groups, dom_gt, text_emb=None):
        super().__init__(cfg, inputs, labels, groups, dom_gt, text_emb)
        assert len(self.text_emb) == len(np.unique(self.domain_labels)), "text embedding and domain labels don't match"

    def __getitem__(self, idx):
        inp, label, group, dom_label = super().__getitem__(idx)
        return inp, label, group, self.text_emb[dom_label]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class ClipMLP(Noop):

    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        super().__init__(text_prompts, model, cfg, neutral_prompts)
        self.text_emb = []
        self.cfg = cfg
        self.uid = uuid.uuid4()

    def create_model(self, inputs):
        B, W  = inputs.shape
        self.model_conf = OmegaConf.create({"in_dim": W, "h_dim": self.cfg.METHOD.MODEL.HIDDEN_DIM, "out_dim": self.train_dataset.num_classes, "num_classes": self.train_dataset.num_classes, "num_domains": self.train_dataset.num_domains, "num_layers": self.cfg.METHOD.MODEL.NUM_LAYERS})
        self.cfg = OmegaConf.merge(self.cfg, self.model_conf)
        net = MLP(self.cfg)
        self.net = net.cuda()
        net = torch.nn.DataParallel(self.net)
        cudnn.benchmark = True

    def create_criterion(self):
        weights = self.train_dataset.class_weights if self.cfg.DATA.UPWEIGHT_CLASSES else None
        if self.model_conf.num_classes == 2 and not self.cfg.METHOD.MODEL.SEPERATE_CLASSES:
            self.class_criterion = nn.BCEWithLogitsLoss()
            self.m = nn.Sigmoid()
        else:
            self.class_criterion = nn.CrossEntropyLoss(weight=weights.cuda())
            self.m = nn.Softmax(dim=1)

    def train_debias(self, inputs, labels, groups, dom_gt, test_inputs, test_labels, test_groups, test_dom_gt):
        """
        Set up data, model, loss, opt and run
        """
        self.train_dataset = EmbeddingDataset(self.cfg, inputs, labels, groups, dom_gt, self.text_emb)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=True)
        self.test_dataset = EmbeddingDataset(self.cfg, test_inputs, test_labels, test_groups, test_dom_gt, self.text_emb)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=False)

        self.create_model(inputs) # create model
        if self.cfg.METHOD.MODEL.RESUME:
            self.load_checkpoint(self.cfg.METHOD.MODEL.CHECKPOINT)
    
        self.optimizer = AdamW(self.net.parameters(), lr=self.cfg.METHOD.MODEL.LR, weight_decay=self.cfg.METHOD.MODEL.WEIGHT_DECAY)
        self.create_criterion() # get loss functions

        if not self.cfg.METHOD.MODEL.RESUME:
            self.best_acc, self.best_epoch = 0, 0
            for epoch in range(self.cfg.EXP.EPOCHS):
                self.train(epoch)
                if self.cfg.EXP.CHECKPOINT_VAL:
                    self.test(epoch, load=False)
            self.save_checkpoint(0.0, epoch, last=True)
            wandb.summary['best val balanced acc'] = self.best_acc
            wandb.summary['best epoch'] = self.best_epoch

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])
        print(f"...loaded checkpoint with acc {checkpoint['acc']}")

    def train(self, epoch):
        print(f"Epoch {epoch}")
        self.net.train()
        train_cls_loss, train_dom_loss, train_loss, cls_correct, total = 0, 0, 0, 0, 0
        for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.train_loader):
            # print(inp, cls_target, cls_group, dom_target)
            inp, cls_target= inp.cuda().float(), cls_target.cuda().long()
            self.optimizer.zero_grad()
            cls_outputs = self.net(inp)
            cls_loss = self.class_criterion(cls_outputs, cls_target)
            loss = cls_loss 
            loss.backward()
            self.optimizer.step()

            train_cls_loss += cls_loss.item()
            train_loss += train_cls_loss + train_dom_loss
            _, cls_predicted = self.m(cls_outputs).max(1)
            total += cls_target.size(0)
            cls_correct += cls_predicted.eq(cls_target).sum().item()

            progress_bar(i, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_cls_loss/(i+1), 100.*cls_correct/total, cls_correct, total))

        wandb.log({"class loss": train_cls_loss/(i+1), "dom loss": train_dom_loss/(i+1), "train cls acc": 100.*cls_correct/total})

    def test(self, epoch, load=True):
        ## load best model
        if load:
            ckpt = f"./checkpoint/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth"
            self.load_checkpoint(ckpt)
        self.net.eval()
        test_cls_loss, test_dom_loss, cls_correct, total = 0, 0, 0, 0
        cls_true, cls_pred, cls_groups, dom_true = np.array([]), np.array([]), np.array([]), np.array([])
        with torch.no_grad():
            for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.test_loader):

                inp, cls_target = inp.cuda().float(), cls_target.cuda().long()
                cls_outputs = self.net(inp)
                cls_loss = self.class_criterion(cls_outputs, cls_target)
                loss = cls_loss 

                test_cls_loss += cls_loss.item()
                _, cls_predicted = self.m(cls_outputs).max(1)
                total += cls_target.size(0)
                cls_correct += cls_predicted.eq(cls_target).sum().item()
                # this is for creating the confusion matrix
                cls_true = np.append(cls_true, cls_target.cpu().numpy())
                cls_pred = np.append(cls_pred, cls_predicted.cpu().numpy())
                cls_groups = np.append(cls_groups, cls_group.cpu().numpy())
                dom_true = np.append(dom_true, dom_target.cpu().numpy())

                progress_bar(i, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_cls_loss/(i+1), 100.*cls_correct/total, cls_correct, total))
        
        accuracy, balanced_acc, class_accuracy, group_accuracy =  evaluate(cls_pred, cls_true, cls_groups)
        wandb.log({"val class loss": test_cls_loss, "val dom loss": test_dom_loss, "val cls acc": accuracy, "val balanced class acc": balanced_acc, 
                    "val class acc": class_accuracy, "val group acc": group_accuracy, "best val balanced acc": self.best_acc})
        if balanced_acc > self.best_acc:
            self.best_acc, self.best_epoch = balanced_acc, epoch
            self.save_checkpoint(balanced_acc, epoch)
    
    def eval(self, inputs, ret_probs=False):
        """ Forward pass for classification. if probs=True, return the softmax prob (for ensambling) """
        try:
            ckpt = f"./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth"
            print(f"loading checkpoint {ckpt}...")
            self.load_checkpoint(ckpt)
        except:
            ckpt = f"./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}-last.pth"
            print(f"loading checkpoint {ckpt}...")
            self.load_checkpoint(ckpt)
        generator = chunks(torch.tensor(inputs).cuda().float(), self.cfg.DATA.BATCH_SIZE)
        preds, probs = np.array([]), []
        for i, inp in enumerate(generator):
            if self.cfg.METHOD.NORMALIZE_OUTPUT:
                cls_outputs = self.net(inp / inp.norm(dim=-1, keepdim=True))
            else:
                cls_outputs = self.net(inp)
            probs.append(self.m(cls_outputs).detach().cpu().numpy())
            _, cls_predicted = self.m(cls_outputs).max(1)
            preds = np.append(preds, cls_predicted.cpu().numpy())
        return preds, np.concatenate(probs, axis=0)

    def save_checkpoint(self, acc, epoch, last=False):
        print(f'Saving checkpoint with acc {acc} to ./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth...')
        state = {
            "acc": acc,
            "epoch": epoch,
            "net": self.net.state_dict()
        }
        checkpoint_dir = '/'.join(f'./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}'.split('/')[:-1])
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if last:
            torch.save(state, f'./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}-last.pth')
            wandb.save(f'./checkpoint/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}-last.pth')
        else:
            # make checkpoint directory and DomainNetMini directory
            torch.save(state, f'./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth')
            wandb.save(f'./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth')

class DirectionLoss(torch.nn.Module):
    """
    Directional Loss taken from StyleGAN Nada paper. Takes in the normalized
    differences between the images and text of the source and target domain
    (aka the "direction" in image and text space) and computes their similarity. 
    source: https://github.com/rinongal/StyleGAN-nada/blob/main/ZSSGAN/criteria/clip_loss.py
    """

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

class AugE2EMLPMulti(ClipMLP):
    """
    End-to-end version of SALEM with the class consistency loss replaced with the 
    classification loss. (Supposedly) Works on multiple domains.
    """

    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        super().__init__(text_prompts, model, cfg, neutral_prompts)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(self.cfg.EXP.CLIP_MODEL, device)
        model.eval()
        text_embs = zeroshot_classifier([[f"a photo of a {c}"] for c in self.class_names], model, model_type=cfg.EXP.IMAGE_FEATURES)
        self.class_text_embs = text_embs.float().cuda()
        self.domain_indexes = []
        for prompt in list(self.cfg.EXP.TEXT_PROMPTS):
            if len(list(self.cfg.AUGMENTATION.DOM_LABELS)) == 0:
                if type(prompt) == omegaconf.listconfig.ListConfig:
                    print("remove list")
                    prompt = prompt[0]
                print(type(prompt), prompt, self.domain_names)
                dom_idx = [i for i in range(len(self.domain_names)) if self.domain_names[i] in prompt]
                assert len(dom_idx) == 1, "error indexing domains, make sure your text prompt contains the name of the domain"
                self.domain_indexes.append(dom_idx[0])
            else:
                self.domain_indexes = [self.domain_names.index(p) for p in list(self.cfg.AUGMENTATION.DOM_LABELS)]
        print("domain indexes ", self.domain_indexes)

        try:
            self.orig_prompts = torch.Tensor(self.get_orig_text_embeddings(self.text_prompts).transpose((1, 0, 2))).float().cuda()
            self.neutral_embs = torch.Tensor(self.get_orig_text_embeddings(self.neutral_prompts).transpose((1, 0, 2))).float().cuda()
            # if len(self.orig_prompts) > 1:
            #     raise ValueError("this only works for one domain shift atm")
            print('=========================')
            print('=========================')
            print("task specific text emb ", self.orig_prompts.shape, self.orig_prompts[0].shape, torch.transpose(self.orig_prompts[0].float().cuda(), 1, 0).shape, self.class_text_embs.shape)
            print('=========================')
            print('=========================')
            # stack = [torch.mean(self.neutral_embs, dim=1)] + torch.mean(self.orig_prompts, dim=1)
            self.val_dom_check = torch.squeeze(torch.cat([torch.mean(self.neutral_embs, dim=1), torch.mean(self.orig_prompts, dim=1)]), dim=1).float().cuda()
            self.val_dom_check = torch.transpose(self.val_dom_check, 0, 1)
            print("val domain check shape ", self.val_dom_check.shape, self.class_text_embs.shape)
        except:
            print("can't load prompts")

    @staticmethod
    def get_class_logits(outputs, class_embs):
        outputs_norm = outputs / outputs.norm(dim=-1, keepdim=True) 
        return torch.matmul(outputs_norm, class_embs)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])
        self.aug_net.load_state_dict(checkpoint['aug_net'])
        print(f"...loaded checkpoint with acc {checkpoint['acc']}")

    def save_checkpoint(self, acc, epoch, last=False):
        print(f'Saving checkpoint with acc {acc} to ./checkpoint/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth...')
        state = {
            "acc": acc,
            "epoch": epoch,
            "net": self.net.state_dict(),
            "aug_net": self.aug_net.state_dict()
        }
        if last:
            torch.save(state, f'./checkpoint/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}-last.pth')
            wandb.save(f'./checkpoint/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}-last.pth')
        else:
            torch.save(state, f'./checkpoint/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth')
            wandb.save(f'./checkpoint/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth')

    def create_model(self, inputs):
        B, W  = inputs.shape
        self.model_conf = OmegaConf.create({"in_dim": W, "h_dim": W, "out_dim": self.train_dataset.num_classes, "num_classes": self.train_dataset.num_classes, "num_domains": self.train_dataset.num_domains, "num_layers": self.cfg.METHOD.MODEL.NUM_LAYERS})
        self.cfg = OmegaConf.merge(self.cfg, self.model_conf)
        net = MLP(self.cfg)
        self.net = net.cuda()
        # net = torch.nn.DataParallel(self.net)
        cudnn.benchmark = True
        self.augmentation_model_conf = OmegaConf.create({"in_dim": W, "h_dim": self.cfg.AUGMENTATION.MODEL.HIDDEN_DIM, "out_dim": W, "num_classes": self.train_dataset.num_classes, "num_layers": self.cfg.AUGMENTATION.MODEL.NUM_LAYERS})
        self.num_domains = len(self.text_prompts)
        aug_net = nn.ModuleList([MLP(OmegaConf.merge(self.cfg, self.augmentation_model_conf)) for i in range(self.num_domains)])
        self.aug_net = aug_net.cuda()
        # self.aug_net = torch.nn.DataParallel(self.aug_net)
        wandb.watch(aug_net, log_freq=10)
        print("NUM DOMAINS ", len(self.text_prompts), self.text_prompts, self.train_dataset.num_domains)

    def create_criterion(self):
        weights = self.train_dataset.class_weights if self.cfg.DATA.UPWEIGHT_CLASSES else None
        if self.model_conf.num_classes == 2 and not self.cfg.METHOD.MODEL.SEPERATE_CLASSES:
            self.class_criterion = nn.BCEWithLogitsLoss()
            self.m = nn.Sigmoid()
        else:
            self.class_criterion = nn.CrossEntropyLoss(weight=weights.cuda())
            self.m = nn.Softmax(dim=1)
        self.domain_criterion = DirectionLoss(self.cfg.AUGMENTATION.LOSS_TYPE)
        self.clip_nn_loss = nn.CrossEntropyLoss()
        self.regularizer = DirectionLoss(self.cfg.AUGMENTATION.LOSS_TYPE)

    def train_debias(self, inputs, labels, groups, dom_gt, test_inputs, test_labels, test_groups, test_dom_gt):
        """
        Set up data, model, loss, opt and run
        """
        self.train_dataset = EmbeddingDataset(self.cfg, inputs, labels, groups, dom_gt, self.text_emb)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=True)
        self.test_dataset = EmbeddingDataset(self.cfg, test_inputs, test_labels, test_groups, test_dom_gt, self.text_emb)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=False)

        self.create_model(inputs) # create model
        if self.cfg.METHOD.MODEL.RESUME:
            self.load_checkpoint(self.cfg.METHOD.MODEL.CHECKPOINT)
    
        self.optimizer = AdamW(self.net.parameters(), lr=self.cfg.METHOD.MODEL.LR, weight_decay=self.cfg.METHOD.MODEL.WEIGHT_DECAY)
        self.aug_optimizer = AdamW(self.aug_net.parameters(), lr=self.cfg.AUGMENTATION.MODEL.LR, weight_decay=self.cfg.AUGMENTATION.MODEL.WEIGHT_DECAY)
        self.create_criterion() # get loss functions

        if not self.cfg.METHOD.MODEL.RESUME:
            self.best_acc, self.best_epoch = 0, 0
            for epoch in range(self.cfg.EXP.EPOCHS):
                self.train(epoch)
                if self.cfg.EXP.CHECKPOINT_VAL:
                    self.test(epoch, load=False)
            self.save_checkpoint(self.best_acc, epoch, last=True)
            wandb.summary['best val balanced acc'] = self.best_acc
            wandb.summary['best epoch'] = self.best_epoch

    def get_orig_text_embeddings(self, prompts):
        if self.cfg.AUGMENTATION.GENERIC:
            text_embs = zeroshot_classifier(prompts, self.model, model_type=cfg.EXP.IMAGE_FEATURES).cpu().numpy().T
            text_embs /= np.linalg.norm(text_embs, axis=-1, keepdims=True)
            return text_embs
        else:
            all_texts, text_names = [], []
            for t in prompts:
                texts = [[t.format(c)] for c in self.class_names]
                text_names += texts
                text_emb = zeroshot_classifier(texts, self.model, model_type=cfg.EXP.IMAGE_FEATURES).cpu().numpy().T
                text_emb /= np.linalg.norm(text_emb, axis=-1, keepdims=True)
                all_texts.append(text_emb)
            # print("array of orig prompts ", text_names, np.array(all_texts).transpose((1, 0, 2)).shape)
            return np.array(all_texts).transpose((1, 0, 2))

    @staticmethod
    def filter(inp, dom_target, domain):
        # print(dom_target[dom_target == domain])
        if len(dom_target[dom_target == domain]) == 0:
            return torch.tensor([])
        return torch.stack([inp[i] for i in range(len(dom_target)) if dom_target[i].item() == domain]).cuda()

    def train(self, epoch):
        print(f"Epoch {epoch}")
        torch.autograd.set_detect_anomaly(True)
        self.net.train()
        self.aug_net.train()
        train_cls_loss, train_dom_loss, train_loss, train_reg_loss, cls_correct, total = 0, 0, 0, 0, 0, 0
        for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.train_loader):
            inp, cls_target= inp.cuda().float(), cls_target.cuda().long()
            self.aug_optimizer.zero_grad()
            self.optimizer.zero_grad()
            if self.cfg.METHOD.NORMALIZE_OUTPUT:
                cls_outputs = self.net(inp / inp.norm(dim=-1, keepdim=True))
            else:
                cls_outputs = self.net(inp)

            img_directional, text_directional = [], []
            out, labels, nn_labels, nn_out = [], [], [], []
            aug_logits, aug_labels = [], []

            if np.random.rand() <= self.cfg.AUGMENTATION.RANDOMIZE_PROB:
                out.append(cls_outputs)
                labels.append(cls_target)

            for domain in range(self.num_domains):
                aug_inp = self.aug_net[domain](inp)
                if self.cfg.METHOD.NORMALIZE_OUTPUT:
                    aug_cls_outputs = self.net(aug_inp / aug_inp.norm(dim=-1, keepdim=True))
                else:
                    aug_cls_outputs = self.net(aug_inp)
                
                if self.cfg.AUGMENTATION.RANDOMIZE:
                    if np.random.rand() > 0.5:
                        out.append(aug_cls_outputs)
                        labels.append(cls_target)
                else:
                    out.append(aug_cls_outputs)
                    labels.append(cls_target)

                # compute directional loss
                if self.cfg.METHOD.NORMALIZE:
                    diff_img_embeddings = torch.sub(aug_inp / aug_inp.norm(dim=-1, keepdim=True), inp)
                else:
                    diff_img_embeddings = torch.sub(aug_inp, inp)
                img_directional.append(diff_img_embeddings / diff_img_embeddings.norm(dim=-1, keepdim=True))
                num_classes, num_domains, emb_dim = self.text_embeddings.shape
                text_emb = torch.Tensor(self.text_embeddings[:, domain, :]).reshape((num_classes, 1, emb_dim)).cuda()

                if not self.cfg.AUGMENTATION.GENERIC:
                    text_diffs = torch.cat([text_emb[y] for y in cls_target])
                else:
                    text_diffs = torch.cat([text_emb for y in cls_target])

                text_directional.append(text_diffs)

                if self.cfg.AUGMENTATION.DOM_SPECIFIC_XE:
                    cls_logits = self.get_class_logits(aug_cls_outputs, torch.transpose(self.orig_prompts[domain].float().cuda(), 1, 0))
                else:
                    cls_logits = self.get_class_logits(aug_inp / aug_inp.norm(dim=-1, keepdim=True), self.class_text_embs)
                # cls_logits = self.get_class_logits(aug_inp / aug_inp.norm(dim=-1, keepdim=True), self.class_text_embs)  
                aug_logits.append(cls_logits)
                aug_labels.append(cls_target) 
            
            clip_cls_loss = self.cfg.AUGMENTATION.CC_WEIGHT * self.class_criterion(torch.cat(aug_logits).cuda(), torch.cat(aug_labels).cuda())
            cls_loss = (1 - self.cfg.AUGMENTATION.ALPHA) * self.class_criterion(torch.cat(out).cuda(), torch.cat(labels).cuda())
            if self.cfg.METHOD.NORMALIZE:
                reg_loss = self.cfg.AUGMENTATION.REG_WEIGHT * self.domain_criterion(inp, aug_inp / aug_inp.norm(dim=-1, keepdim=True)).mean()
            else:
                reg_loss = self.cfg.AUGMENTATION.REG_WEIGHT * self.domain_criterion(inp, aug_inp).mean()
                
            domain_loss = self.cfg.AUGMENTATION.DOM_WEIGHT * self.cfg.AUGMENTATION.ALPHA * self.domain_criterion(torch.cat(img_directional).cuda(), torch.cat(text_directional).cuda()).mean()
            # breakpoint()

            loss = cls_loss + domain_loss + reg_loss + clip_cls_loss
            loss.backward()
            self.aug_optimizer.step()
            self.optimizer.step()

            train_cls_loss += cls_loss.item()
            train_dom_loss += domain_loss.item()
            train_loss += train_cls_loss + train_dom_loss
            train_reg_loss += reg_loss.item()
            _, cls_predicted = self.m(torch.cat(out)).max(1)
            total += torch.cat(labels).size(0)
            cls_correct += cls_predicted.eq(torch.cat(labels)).sum().item()

            progress_bar(i, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_cls_loss/(i+1), 100.*cls_correct/total, cls_correct, total))

        wandb.log({"train loss": train_loss/(i+1), "class loss": train_cls_loss/(i+1), "domain loss": train_dom_loss/(i+1), "train cls acc": 100.*cls_correct/total, "reg loss": train_reg_loss/(i+1)})

    def test(self, epoch, load=True):
        ## load best model
        if load:
            ckpt = f"./checkpoint/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth"
            self.load_checkpoint(ckpt)
        self.net.eval()
        self.aug_net.eval()
        test_cls_loss, test_dom_loss, cls_correct, total = 0, 0, 0, 0
        cls_true, cls_pred, cls_groups, dom_true = np.array([]), np.array([]), np.array([]), np.array([])


        with torch.no_grad():
            for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.test_loader):
                img_directional, text_directional = [], []
                out, labels, groups = [], [], []

                inp, cls_target = inp.cuda().float(), cls_target.cuda().long()
                if self.cfg.METHOD.NORMALIZE_OUTPUT:
                    cls_outputs = self.net(inp / inp.norm(dim=-1, keepdim=True))
                else:
                    cls_outputs = self.net(inp)
                out.append(cls_outputs)
                labels.append(cls_target)
                groups.append(cls_group)

                for domain in range(self.num_domains):
                    aug_inp = self.aug_net[domain](inp)
                    if self.cfg.METHOD.NORMALIZE_OUTPUT:
                        aug_outputs = self.net(aug_inp / aug_inp.norm(dim=-1, keepdim=True))
                    else:
                        aug_outputs = self.net(aug_inp)
                    out.append(aug_outputs)
                    labels.append(cls_target)
                    groups.append(cls_group)

                    if self.cfg.METHOD.NORMALIZE:
                        diff_img_embeddings = torch.sub(aug_inp / aug_inp.norm(dim=-1, keepdim=True), inp)
                    else:
                        diff_img_embeddings = torch.sub(aug_inp, inp)
                    img_directional.append(diff_img_embeddings / diff_img_embeddings.norm(dim=-1, keepdim=True))
                    num_classes, num_domains, emb_dim = self.text_embeddings.shape
                    text_emb = torch.Tensor(self.text_embeddings[:, domain, :]).reshape((num_classes, 1, emb_dim)).cuda()

                    if not self.cfg.AUGMENTATION.GENERIC:
                        text_diffs = torch.cat([text_emb[y] for y in cls_target])
                    else:
                        text_diffs = torch.cat([text_emb for y in cls_target])

                    text_directional.append(text_diffs)
                    
                cls_loss = (1 - self.cfg.AUGMENTATION.ALPHA) * self.class_criterion(torch.cat(out).cuda(), torch.cat(labels).cuda())
                domain_loss = self.cfg.AUGMENTATION.DOM_WEIGHT * self.cfg.AUGMENTATION.ALPHA * self.domain_criterion(torch.cat(img_directional).cuda(), torch.cat(text_directional).cuda()).mean()

                test_cls_loss += cls_loss.item()
                test_dom_loss += domain_loss.item()
                _, cls_predicted = self.m(torch.cat(out).cuda()).max(1)
                total += cls_target.size(0)

                cls_correct += cls_predicted.eq(torch.cat(labels).cuda()).sum().item()
                # this is for creating the confusion matrix
                cls_true = np.append(cls_true, torch.cat(labels).cpu().numpy())
                cls_pred = np.append(cls_pred, cls_predicted.cpu().numpy())
                cls_groups = np.append(cls_groups, torch.cat(groups).cpu().numpy())
                # dom_true = np.append(dom_true, dom_target.cpu().numpy())

                progress_bar(i, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_cls_loss/(i+1), 100.*cls_correct/total, cls_correct, total))
        
        accuracy, balanced_acc, class_accuracy, group_accuracy =  evaluate(cls_pred, cls_true, cls_groups)
        wandb.log({"val class loss": test_cls_loss, "val dom loss": test_dom_loss, "val cls acc": accuracy, "val balanced class acc": balanced_acc, 
                    "val class acc": class_accuracy, "val group acc": group_accuracy, "best val balanced acc": self.best_acc})
        if balanced_acc > self.best_acc:
            self.best_acc, self.best_epoch = balanced_acc, epoch
            self.save_checkpoint(balanced_acc, epoch)

    def augment_dataset(self, image_features, labels, domain_labels, filenames):
        """
        Used to check the quality of augmentation network. Computes the augmented embeddings
        and changes the domain (rn only works with 1 domain) to be used in the nearest neighbor
        ablations. 
        """
        ckpt = f"./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth"
        print(f"loading checkpoint {ckpt}...")
        self.load_checkpoint(ckpt)
        print("domain indices", self.domain_indexes)

        self.aug_net.eval()
        augmented_features = []
        augmented_labels = []
        augmented_domain_labels = []
        augmented_filenames = []
        for inp, label, domain, filename in zip(image_features, labels, domain_labels, filenames):
            for d in range(self.num_domains):
                aug_inp = self.aug_net[d](torch.unsqueeze(torch.Tensor(inp).float().cuda(), dim=0))
                aug_inp /= aug_inp.norm(dim=-1, keepdim=True)
                augmented_features += [aug_inp.detach().cpu().numpy()[0]]
                augmented_labels += [label]
                augmented_domain_labels += [self.domain_indexes[domain]]
                augmented_filenames += [filename]

        return np.array(augmented_features), np.array(augmented_labels), np.array(augmented_domain_labels), np.array(augmented_filenames)   

class AugE2EBiasMLP(AugE2EMLPMulti):
    """
    E2E version of BiasDirectional
    """
    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        super().__init__(text_prompts, model, cfg, neutral_prompts)
        # if self.cfg.AUGMENTATION.DOM_SPECIFIC_XE:
        # self.orig_prompts = torch.Tensor(self.get_orig_text_embeddings(self.text_prompts).transpose((1, 0, 2))).float().cuda()
        # print("orig prompts shape ", self.orig_prompts.shape)
        #     self.text_features = torch.mean(self.orig_prompts, dim=1)
        #     text_features = self.text_features
        # else:
        self.text_features = torch.tensor(self.orig_prompts).float().cuda()
        print("text features shape ", self.text_features.shape)
        if len(self.text_features.shape) == 3:
            self.dom_text_features = torch.mean(self.text_features, dim=1)
        else:
            self.dom_text_features = self.text_features
        print("NEW text features shape ", self.text_features.shape)
        if self.cfg.AUGMENTATION.DOM_SPECIFIC_XE:
            self.class_text_embs = self.orig_prompts
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load(self.cfg.EXP.CLIP_MODEL, device)
            model.eval()
            text_embs = zeroshot_classifier([[f"a photo of the number {c}"] for c in self.class_names], model, model_type=cfg.EXP.IMAGE_FEATURES)
            self.class_text_embs = text_embs.float().cuda()

    def create_model(self, inputs):
        B, W  = inputs.shape
        self.model_conf = OmegaConf.create({"in_dim": W, "h_dim": W, "out_dim": self.train_dataset.num_classes, "num_classes": self.train_dataset.num_classes, "num_domains": self.train_dataset.num_domains, "num_layers": self.cfg.METHOD.MODEL.NUM_LAYERS})
        self.cfg = OmegaConf.merge(self.cfg, self.model_conf)
        net = MLP(self.cfg)
        self.net = net.cuda()
        # net = torch.nn.DataParallel(self.net)
        cudnn.benchmark = True
        self.augmentation_model_conf = OmegaConf.create({"in_dim": W, "h_dim": self.cfg.AUGMENTATION.MODEL.HIDDEN_DIM, "out_dim": W, "num_classes": self.train_dataset.num_classes, "num_layers": self.cfg.AUGMENTATION.MODEL.NUM_LAYERS})
        self.num_domains = len(self.text_prompts)
        aug_net = nn.ModuleList([MLP(OmegaConf.merge(self.cfg, self.augmentation_model_conf))])
        self.aug_net = aug_net.cuda()
        # self.aug_net = torch.nn.DataParallel(self.aug_net)
        wandb.watch(aug_net, log_freq=10)
        print("NUM DOMAINS ", len(self.text_prompts), self.text_prompts, self.train_dataset.num_domains)

    def train_debias(self, inputs, labels, groups, dom_gt, test_inputs, test_labels, test_groups, test_dom_gt):
        """
        Set up data, model, loss, opt and run
        """
        self.train_dataset = EmbeddingDataset(self.cfg, inputs, labels, groups, dom_gt, text_emb=self.dom_text_features)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=True)
        self.test_dataset = EmbeddingDataset(self.cfg, test_inputs, test_labels, test_groups, test_dom_gt, text_emb=self.dom_text_features)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=False)

        self.create_model(inputs) # create model
        if self.cfg.METHOD.MODEL.RESUME:
            self.load_checkpoint(self.cfg.METHOD.MODEL.CHECKPOINT)
    
        self.optimizer = AdamW(self.net.parameters(), lr=self.cfg.METHOD.MODEL.LR, weight_decay=self.cfg.METHOD.MODEL.WEIGHT_DECAY)
        self.aug_optimizer = AdamW(self.aug_net.parameters(), lr=self.cfg.AUGMENTATION.MODEL.LR, weight_decay=self.cfg.AUGMENTATION.MODEL.WEIGHT_DECAY)
        self.create_criterion() # get loss functions

        if not self.cfg.METHOD.MODEL.RESUME:
            self.best_acc, self.best_epoch = 0, 0
            for epoch in range(self.cfg.EXP.EPOCHS):
                self.train(epoch)
                if self.cfg.EXP.CHECKPOINT_VAL:
                    self.test(epoch, load=False)
            self.save_checkpoint(self.best_acc, epoch, last=True)
            wandb.summary['best val balanced acc'] = self.best_acc
            wandb.summary['best epoch'] = self.best_epoch

    @staticmethod
    def get_inv(label):
        return 1 if label == 0 else 0
    
    @staticmethod
    def filter(inp, dom_target, domain):
        # print(dom_target[dom_target == domain])
        if len(dom_target[dom_target == domain]) == 0:
            return torch.tensor([])
        return torch.stack([inp[i] for i in range(len(dom_target)) if dom_target[i].item() == domain]).cuda()

    def train(self, epoch):
        print(f"Epoch {epoch}")
        torch.autograd.set_detect_anomaly(True)
        self.net.train()
        self.aug_net.train()
        train_cls_loss, train_dom_loss, train_loss, train_reg_loss, cls_correct, total = 0, 0, 0, 0, 0, 0
        for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.train_loader):
            inp, cls_target = inp.cuda().float(), cls_target.cuda().long()
            self.aug_optimizer.zero_grad()
            self.optimizer.zero_grad()
            if self.cfg.METHOD.NORMALIZE_OUTPUT:
                cls_outputs = self.net(inp / inp.norm(dim=-1, keepdim=True))
            else:
                cls_outputs = self.net(inp)

            img_directional, text_directional = [], []
            out, labels, nn_labels, nn_out = [], [], [], []
            aug_logits, aug_labels = [], []
            cycle_gan_inp, cycle_aug_inp = [], []

            if np.random.rand() <= self.cfg.AUGMENTATION.RANDOMIZE_PROB:
                out.append(cls_outputs)
                labels.append(cls_target)

            for domain in range(len(self.aug_net)):
                aug_inp = self.aug_net[domain](inp)
                # dom_inp = self.filter(inp, dom_target, domain)
                # dom_lab = self.filter(cls_target, dom_target, domain)
                # dom_aug_inp = self.filter(aug_inp, dom_target, domain) 
                # if len(dom_aug_inp) == 0:
                #     continue
                if self.cfg.METHOD.NORMALIZE_OUTPUT:
                    aug_cls_outputs = self.net(aug_inp / aug_inp.norm(dim=-1, keepdim=True))
                else:
                    aug_cls_outputs = self.net(aug_inp)

                # dom_aug_out = self.filter(aug_cls_outputs, dom_target, domain)
                
                if self.cfg.AUGMENTATION.RANDOMIZE:
                    if np.random.rand() > 0.5:
                        out.append(aug_cls_outputs)
                        labels.append(cls_target)
                else:
                    out.append(aug_cls_outputs)
                    labels.append(cls_target)

                # compute directional loss
                if self.cfg.METHOD.NORMALIZE:
                    diff_img_embeddings = torch.sub(aug_inp / aug_inp.norm(dim=-1, keepdim=True), inp)
                else:
                    diff_img_embeddings = torch.sub(aug_inp, inp)
                img_directional.append(diff_img_embeddings / diff_img_embeddings.norm(dim=-1, keepdim=True))
                num_classes, num_domains, emb_dim = self.text_features.shape
                text_emb = torch.tensor(self.text_features).cuda()

                if not self.cfg.AUGMENTATION.GENERIC:
                    text_diffs = torch.stack([torch.sub(text_emb[self.get_inv(d)][y], text_emb[d][y]) for d, y in zip(dom_target, cls_target)])
                else:
                    text_diffs = torch.stack([torch.sub(text_emb[self.get_inv(d)], text_emb[d]) for d in dom_target])

                text_directional.append(text_diffs)

                if self.cfg.AUGMENTATION.DOM_SPECIFIC_XE:
                    cls_logits = self.get_class_logits(aug_inp, torch.transpose(self.orig_prompts[self.get_inv(domain)].float().cuda(), 1, 0))
                else:
                    cls_logits = self.get_class_logits(aug_inp / aug_inp.norm(dim=-1, keepdim=True), self.class_text_embs)
                # cls_logits = self.get_class_logits(aug_inp / aug_inp.norm(dim=-1, keepdim=True), self.class_text_embs)  
                aug_logits.append(cls_logits)
                aug_labels.append(cls_target) 
            
            clip_cls_loss = self.cfg.AUGMENTATION.CC_WEIGHT * self.class_criterion(torch.cat(aug_logits).cuda(), torch.cat(aug_labels).cuda())
            cls_loss = (1 - self.cfg.AUGMENTATION.ALPHA) * self.class_criterion(torch.cat(out).cuda(), torch.cat(labels).cuda())
            if self.cfg.METHOD.NORMALIZE:
                reg_loss = self.cfg.AUGMENTATION.REG_WEIGHT * self.domain_criterion(inp, aug_inp / aug_inp.norm(dim=-1, keepdim=True)).mean()
            else:
                reg_loss = self.cfg.AUGMENTATION.REG_WEIGHT * self.domain_criterion(inp, aug_inp).mean()
                
            domain_loss = self.cfg.AUGMENTATION.DOM_WEIGHT * self.cfg.AUGMENTATION.ALPHA * self.domain_criterion(torch.cat(img_directional).cuda(), torch.cat(text_directional).cuda()).mean()
            # breakpoint()

            loss = cls_loss + domain_loss + reg_loss + clip_cls_loss
            loss.backward()
            self.aug_optimizer.step()
            self.optimizer.step()

            train_cls_loss += cls_loss.item()
            train_dom_loss += domain_loss.item()
            train_loss += train_cls_loss + train_dom_loss
            train_reg_loss += reg_loss.item()
            _, cls_predicted = self.m(torch.cat(out)).max(1)
            total += torch.cat(labels).size(0)
            cls_correct += cls_predicted.eq(torch.cat(labels)).sum().item()

            progress_bar(i, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_cls_loss/(i+1), 100.*cls_correct/total, cls_correct, total))

        wandb.log({"train loss": train_loss/(i+1), "class loss": train_cls_loss/(i+1), "domain loss": train_dom_loss/(i+1), "train cls acc": 100.*cls_correct/total, "reg loss": train_reg_loss/(i+1)})

    def test(self, epoch, load=True):
        ## load best model
        if load:
            ckpt = f"./checkpoint/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth"
            self.load_checkpoint(ckpt)
        self.net.eval()
        self.aug_net.eval()
        test_cls_loss, test_dom_loss, cls_correct, total = 0, 0, 0, 0
        cls_true, cls_pred, cls_groups, dom_true = np.array([]), np.array([]), np.array([]), np.array([])


        with torch.no_grad():
            for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.test_loader):
                img_directional, text_directional = [], []
                out, labels, groups = [], [], []
                aug_logits, aug_labels = [], []

                inp, cls_target = inp.cuda().float(), cls_target.cuda().long()
                if self.cfg.METHOD.NORMALIZE_OUTPUT:
                    cls_outputs = self.net(inp / inp.norm(dim=-1, keepdim=True))
                else:
                    cls_outputs = self.net(inp)
                out.append(cls_outputs)
                labels.append(cls_target)
                groups.append(cls_group)

                for domain in range(len(self.aug_net)):
                    aug_inp = self.aug_net[domain](inp)
                    # if self.cfg.METHOD.NORMALIZE_OUTPUT:
                    #     aug_outputs = self.net(aug_inp / aug_inp.norm(dim=-1, keepdim=True))
                    # else:
                    #     aug_outputs = self.net(aug_inp)
                    if self.cfg.METHOD.NORMALIZE_OUTPUT:
                        aug_cls_outputs = self.net(aug_inp / aug_inp.norm(dim=-1, keepdim=True))
                    else:
                        aug_cls_outputs = self.net(aug_inp)
                
                    out.append(aug_cls_outputs)
                    labels.append(cls_target)
                    groups.append(cls_group)

                    # compute directional loss
                    if self.cfg.METHOD.NORMALIZE:
                        diff_img_embeddings = torch.sub(aug_inp / aug_inp.norm(dim=-1, keepdim=True), inp)
                    else:
                        diff_img_embeddings = torch.sub(aug_inp, inp)
                    img_directional.append(diff_img_embeddings / diff_img_embeddings.norm(dim=-1, keepdim=True))
                    num_classes, num_domains, emb_dim = self.text_features.shape
                    text_emb = torch.tensor(self.text_features).cuda()

                    if not self.cfg.AUGMENTATION.GENERIC:
                        text_diffs = torch.stack([torch.sub(text_emb[self.get_inv(d)][y], text_emb[d][y]) for d, y in zip(dom_target, cls_target)])
                    else:
                        text_diffs = torch.stack([torch.sub(text_emb[self.get_inv(d)], text_emb[d]) for d in dom_target])

                    text_directional.append(text_diffs)

                    if self.cfg.AUGMENTATION.DOM_SPECIFIC_XE:
                        cls_logits = self.get_class_logits(aug_inp, torch.transpose(self.orig_prompts[self.get_inv(domain)].float().cuda(), 1, 0))
                    else:
                        cls_logits = self.get_class_logits(aug_inp / aug_inp.norm(dim=-1, keepdim=True), self.class_text_embs)
                    # cls_logits = self.get_class_logits(aug_inp / aug_inp.norm(dim=-1, keepdim=True), self.class_text_embs)  
                    aug_logits.append(cls_logits)
                    aug_labels.append(cls_target) 
                    
                # cls_loss = self.cfg.AUGMENTATION.CC_WEIGHT * self.class_criterion(torch.cat(aug_logits).cuda(), torch.cat(aug_labels).cuda())
                cls_loss = (1 - self.cfg.AUGMENTATION.ALPHA) * self.class_criterion(torch.cat(out).cuda(), torch.cat(labels).cuda())
                domain_loss = self.cfg.AUGMENTATION.DOM_WEIGHT * self.cfg.AUGMENTATION.ALPHA * self.domain_criterion(torch.cat(img_directional).cuda(), torch.cat(text_directional).cuda()).mean()

                test_cls_loss += cls_loss.item()
                test_dom_loss += domain_loss.item()
                _, cls_predicted = self.m(torch.cat(out).cuda()).max(1)
                total += cls_target.size(0)

                cls_correct += cls_predicted.eq(torch.cat(labels).cuda()).sum().item()
                # this is for creating the confusion matrix
                cls_true = np.append(cls_true, torch.cat(labels).cpu().numpy())
                cls_pred = np.append(cls_pred, cls_predicted.cpu().numpy())
                cls_groups = np.append(cls_groups, torch.cat(groups).cpu().numpy())
                # dom_true = np.append(dom_true, dom_target.cpu().numpy())

                progress_bar(i, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_cls_loss/(i+1), 100.*cls_correct/total, cls_correct, total))
        
        accuracy, balanced_acc, class_accuracy, group_accuracy =  evaluate(cls_pred, cls_true, cls_groups)
        wandb.log({"val class loss": test_cls_loss, "val dom loss": test_dom_loss, "val cls acc": accuracy, "val balanced class acc": balanced_acc, 
                    "val class acc": class_accuracy, "val group acc": group_accuracy, "best val balanced acc": self.best_acc})
        if balanced_acc > self.best_acc:
            self.best_acc, self.best_epoch = balanced_acc, epoch
            self.save_checkpoint(balanced_acc, epoch)

    def augment_dataset(self, image_features, labels, domain_labels, filenames):
        """
        Used to check the quality of augmentation network. Computes the augmented embeddings
        and changes the domain (rn only works with 1 domain) to be used in the nearest neighbor
        ablations. 
        """
        ckpt = f"./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth"
        print(f"loading checkpoint {ckpt}...")
        self.load_checkpoint(ckpt)
        print("domain indices", self.domain_indexes, self.domain_names)

        self.aug_net.eval()
        augmented_features = []
        augmented_labels = []
        augmented_domain_labels = []
        augmented_filenames = []
        for inp, label, domain, filename in zip(image_features, labels, domain_labels, filenames):
            aug_inp = self.aug_net[0](torch.unsqueeze(torch.Tensor(inp).float().cuda(), dim=0))
            aug_inp /= aug_inp.norm(dim=-1, keepdim=True)
            augmented_features += [aug_inp.detach().cpu().numpy()[0]]
            augmented_labels += [label]
            augmented_domain_labels += [self.get_inv(domain)]
            augmented_filenames += [filename]

        return np.array(augmented_features), np.array(augmented_labels), np.array(augmented_domain_labels), np.array(augmented_filenames)   

class AugE2EBiasMLPNew(AugE2EBiasMLP):
    
    def train(self, epoch):
        print(f"Epoch {epoch}")
        self.net.train()
        self.aug_net.train()
        train_cls_loss, train_dom_loss, train_loss, train_reg_loss, cls_correct, total = 0, 0, 0, 0, 0, 0
        for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.train_loader):
            inp, cls_target= inp.cuda().float(), cls_target.cuda().long()
            self.aug_optimizer.zero_grad()
            self.optimizer.zero_grad()
            if self.cfg.METHOD.NORMALIZE_OUTPUT:
                cls_outputs = self.net(inp / inp.norm(dim=-1, keepdim=True))
            else:
                cls_outputs = self.net(inp)

            img_directional, text_directional = [], []
            out, labels, nn_labels, nn_out = [], [], [], []
            aug_logits, aug_labels = [], []

            if np.random.rand() <= self.cfg.AUGMENTATION.RANDOMIZE_PROB:
                out.append(cls_outputs)
                labels.append(cls_target)

            for domain in range(self.num_domains):
                aug_inp = self.aug_net[domain](inp)
                if self.cfg.METHOD.NORMALIZE_OUTPUT:
                    aug_cls_outputs = self.net(aug_inp / aug_inp.norm(dim=-1, keepdim=True))
                else:
                    aug_cls_outputs = self.net(aug_inp)
                
                if self.cfg.AUGMENTATION.RANDOMIZE:
                    if np.random.rand() > 0.5:
                        out.append(aug_cls_outputs)
                        labels.append(cls_target)
                else:
                    out.append(aug_cls_outputs)
                    labels.append(cls_target)

                # compute directional loss
                if self.cfg.METHOD.NORMALIZE:
                    diff_img_embeddings = torch.sub(aug_inp / aug_inp.norm(dim=-1, keepdim=True), inp)
                else:
                    diff_img_embeddings = torch.sub(aug_inp, inp)
                img_directional.append(diff_img_embeddings / diff_img_embeddings.norm(dim=-1, keepdim=True))
                num_classes, num_domains, emb_dim = self.text_embeddings.shape
                text_emb = torch.Tensor(self.text_embeddings[:, domain, :]).reshape((num_classes, 1, emb_dim)).cuda()

                if not self.cfg.AUGMENTATION.GENERIC:
                    text_diffs = torch.cat([text_emb[y] for y in cls_target])
                else:
                    text_diffs = torch.cat([text_emb for y in cls_target])

                text_directional.append(text_diffs)

                if self.cfg.AUGMENTATION.DOM_SPECIFIC_XE:
                    cls_logits = self.get_class_logits(aug_cls_outputs, torch.transpose(self.orig_prompts[domain].float().cuda(), 1, 0))
                else:
                    cls_logits = self.get_class_logits(aug_inp / aug_inp.norm(dim=-1, keepdim=True), self.class_text_embs)
                # cls_logits = self.get_class_logits(aug_inp / aug_inp.norm(dim=-1, keepdim=True), self.class_text_embs)  
                aug_logits.append(cls_logits)
                aug_labels.append(cls_target) 
            
            clip_cls_loss = self.cfg.AUGMENTATION.CC_WEIGHT * self.class_criterion(torch.cat(aug_logits).cuda(), torch.cat(aug_labels).cuda())
            cls_loss = (1 - self.cfg.AUGMENTATION.ALPHA) * self.class_criterion(torch.cat(out).cuda(), torch.cat(labels).cuda())
            if self.cfg.METHOD.NORMALIZE:
                reg_loss = self.cfg.AUGMENTATION.REG_WEIGHT * self.domain_criterion(inp, aug_inp / aug_inp.norm(dim=-1, keepdim=True)).mean()
            else:
                reg_loss = self.cfg.AUGMENTATION.REG_WEIGHT * self.domain_criterion(inp, aug_inp).mean()
                
            domain_loss = self.cfg.AUGMENTATION.DOM_WEIGHT * self.cfg.AUGMENTATION.ALPHA * self.domain_criterion(torch.cat(img_directional).cuda(), torch.cat(text_directional).cuda()).mean()
            # breakpoint()

            loss = cls_loss + domain_loss + reg_loss + clip_cls_loss
            loss.backward()
            self.aug_optimizer.step()
            self.optimizer.step()

            train_cls_loss += cls_loss.item()
            train_dom_loss += domain_loss.item()
            train_loss += train_cls_loss + train_dom_loss
            train_reg_loss += reg_loss.item()
            _, cls_predicted = self.m(torch.cat(out)).max(1)
            total += torch.cat(labels).size(0)
            cls_correct += cls_predicted.eq(torch.cat(labels)).sum().item()

            progress_bar(i, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_cls_loss/(i+1), 100.*cls_correct/total, cls_correct, total))

        wandb.log({"train loss": train_loss/(i+1), "class loss": train_cls_loss/(i+1), "domain loss": train_dom_loss/(i+1), "train cls acc": 100.*cls_correct/total, "reg loss": train_reg_loss/(i+1)})


class MLPDebias(ClipMLP):
    """
    Train DANN version where the source and target are weakly labeled domains
    """
    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        super().__init__(text_prompts, model, cfg, neutral_prompts)
        self.text_emb = zeroshot_classifier(text_prompts, model, model_type=cfg.EXP.IMAGE_FEATURES).T # translates text-> embedding
    
    def create_model(self, inputs):
        B, W  = inputs.shape
        self.model_conf = OmegaConf.create({"in_dim": W, "h_dim": W, "out_dim": W, "num_classes": self.train_dataset.num_classes, "num_domains": self.train_dataset.num_domains, "num_layers": self.cfg.METHOD.MODEL.NUM_LAYERS})
        self.cfg = OmegaConf.merge(self.cfg, self.model_conf)
        self.net = EmbeddingDebiasModel(self.cfg)
        self.net = self.net.cuda()
        net = torch.nn.DataParallel(self.net)
        cudnn.benchmark = True

    def create_criterion(self):
        weights = self.train_dataset.class_weights.cuda() if self.cfg.DATA.UPWEIGHT_CLASSES else None
        dom_weights = self.train_dataset.dom_weights.cuda() if self.cfg.DATA.UPWEIGHT_DOMAINS else None
        if self.model_conf.num_classes == 2 and not self.cfg.METHOD.MODEL.SEPERATE_CLASSES:
            self.class_criterion = nn.BCEWithLogitsLoss()
            self.dom_criterion = nn.BCEWithLogitsLoss()
            self.m = nn.Sigmoid()
        else:
            self.class_criterion = nn.CrossEntropyLoss(weight=weights)
            self.dom_criterion = nn.CrossEntropyLoss(weight=dom_weights)
            self.m = nn.Softmax(dim=1)
            if self.cfg.METHOD.MODEL.WEAK_LABELS: # KL is used instead of CE because pytorch CE doesnt support soft labels
                print(".. using soft labels")
                self.dom_criterion = nn.KLDivLoss(reduction="batchmean")
                self.m = nn.Softmax(dim=1)
            else:
                self.dom_criterion = nn.CrossEntropyLoss(weight=dom_weights)
                self.m = nn.Softmax(dim=1)

    def train(self, epoch):
        print(f"Epoch {epoch}")
        self.net.train()
        train_cls_loss, train_dom_loss, train_loss, cls_correct, dom_correct, total = 0, 0, 0, 0, 0, 0
        num_epochs = self.cfg.EXP.EPOCHS
        len_train_loader = len(self.train_loader)
        for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.train_loader):
            p = float(i + epoch * len_train_loader) / num_epochs / len_train_loader
            alpha = (2. / (1. + np.exp(-10 * p)) - 1) * self.cfg.METHOD.MODEL.DOM_WEIGHT

            inp, cls_target, dom_target = inp.cuda().float(), cls_target.cuda().long(), dom_target.cuda().long()
            self.optimizer.zero_grad()
            cls_outputs, dom_outputs = self.net(inp, alpha = alpha)
            cls_loss = self.class_criterion(cls_outputs, cls_target)
            if self.cfg.METHOD.MODEL.WEAK_LABELS:
                target = torch.full(dom_outputs.shape, 1/self.train_dataset.num_domains).cuda()
                dom_loss = cross_entropy_with_probs(dom_outputs, target) * self.cfg.METHOD.MODEL.DOM_WEIGHT
            else:
                dom_loss = self.dom_criterion(dom_outputs, dom_target)
            loss = cls_loss + dom_loss
            loss.backward()
            self.optimizer.step()

            train_cls_loss += cls_loss.item()
            train_dom_loss += dom_loss.item()
            train_loss += train_cls_loss + train_dom_loss
            _, cls_predicted = self.m(cls_outputs).max(1)
            _, dom_predicted = self.m(dom_outputs).max(1)
            total += cls_target.size(0)
            cls_correct += cls_predicted.eq(cls_target).sum().item()
            dom_correct += dom_predicted.eq(dom_target).sum().item()

            progress_bar(i, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Domain Acc: %.3f%% (%d/%d)'
                         % (train_cls_loss/(i+1), 100.*cls_correct/total, cls_correct, total, 100.*dom_correct/total, dom_correct, total))

        wandb.log({"class loss": train_cls_loss/(i+1), "dom loss": train_dom_loss/(i+1), "train cls acc": 100.*cls_correct/total, "train dom acc": 100.*dom_correct/total})

    def test(self, epoch, load=False):
        self.net.eval()
        test_cls_loss, test_dom_loss, cls_correct, dom_correct, total = 0, 0, 0, 0, 0
        cls_true, cls_pred, cls_groups, dom_true, dom_pred = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        with torch.no_grad():
            for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.test_loader):
                inp, cls_target, dom_target = inp.cuda().float(), cls_target.cuda().long(), dom_target.cuda().long()
                cls_outputs, dom_outputs = self.net(inp)
                cls_loss = self.class_criterion(cls_outputs, cls_target)
                if self.cfg.METHOD.MODEL.WEAK_LABELS:
                    target = torch.full(dom_outputs.shape, 1/self.train_dataset.num_domains).cuda()
                    dom_loss = cross_entropy_with_probs(dom_outputs, target)
                else:
                    dom_loss = self.dom_criterion(dom_outputs, dom_target)
                loss = cls_loss + dom_loss

                test_cls_loss += cls_loss.item()
                test_dom_loss += dom_loss.item()
                _, cls_predicted = self.m(cls_outputs).max(1)
                _, dom_predicted = self.m(dom_outputs).max(1)
                total += cls_target.size(0)
                cls_correct += cls_predicted.eq(cls_target).sum().item()
                dom_correct += dom_predicted.eq(dom_target).sum().item()
                # this is for creating the confusion matrix
                cls_true = np.append(cls_true, cls_target.cpu().numpy())
                cls_pred = np.append(cls_pred, cls_predicted.cpu().numpy())
                cls_groups = np.append(cls_groups, cls_group.cpu().numpy())
                dom_true = np.append(dom_true, dom_target.cpu().numpy())
                dom_pred = np.append(dom_pred, dom_predicted.cpu().numpy())

                progress_bar(i, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Domain Acc: %.3f%% (%d/%d)'
                            % (test_cls_loss/(i+1), 100.*cls_correct/total, cls_correct, total, 100.*dom_correct/total, dom_correct, total))
        
        accuracy, balanced_acc, class_accuracy, group_accuracy =  evaluate(cls_pred, cls_true, cls_groups)
        dom_accuracy, dom_balanced_acc, dom_class_accuracy =  evaluate(dom_pred, dom_true)
        wandb.log({"val class loss": test_cls_loss/(i+1), "val dom loss": test_dom_loss/(i+1), "val cls acc": accuracy, "val balanced class acc": balanced_acc, 
                    "val class acc": class_accuracy, "val dom acc": dom_accuracy, "val balanced dom acc": dom_balanced_acc, 
                    "val dom class acc": dom_class_accuracy, "val group acc": group_accuracy})
        if balanced_acc > self.best_acc:
            self.best_acc, self.best_epoch = balanced_acc, epoch
            self.save_checkpoint(balanced_acc, epoch)

    def eval(self, inputs):
        """ Farward pass for classification """
        ckpt = f"./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}.pth"
        print(f"loading checkpoint {ckpt}...")
        generator = chunks(torch.tensor(inputs).cuda().float(), self.cfg.DATA.BATCH_SIZE)
        preds = np.array([])
        for i, inp in enumerate(generator):
            cls_outputs, _ = self.net(inp)
            _, cls_predicted = self.m(cls_outputs).max(1)
            preds = np.append(preds, cls_predicted.cpu().numpy())
        return preds
