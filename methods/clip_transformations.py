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

try:
    from progress_bar import progress_bar
except:
    progress_bar = lambda current, total, msg: None

import uuid

from methods.predictors import MLP, MPLZS
from clip_utils import zeroshot_classifier, evaluate, get_domain_text_embs

import helpers.text_templates
from helpers.text_templates import imagenet_templates, part_templates, imagenet_templates_small
from helpers.data_helpers import DATASET_CLASSES, DATASET_DOMAINS
import helpers

device = "cuda" if torch.cuda.is_available() else "cpu"

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
class Base:
    """
    Base class for all methods.
    Computes CLIP embeddings. If NEUTRAL_PROMPTS is not empty, then we take the difference between our prompts and the neutral prompt.
    If EXP.GENERAL=False, then we compute the embeddings for each class (num_text_prompts, )
    """
    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        self.class_names = DATASET_CLASSES[cfg.DATA.DATASET]
        self.domain_names = DATASET_DOMAINS[cfg.DATA.DATASET]
        print("TEXT PROMPTS ", text_prompts, neutral_prompts)
        self.target_prompts = text_prompts
        self.cfg = cfg
        self.model = model
        self.neutral_prompts = neutral_prompts
        source_embeddings, target_embeddings = get_domain_text_embs(self.model, cfg, self.neutral_prompts, self.target_prompts, self.class_names)
        # target_embeddings is size (num_domains, num_classes, emb_size)
        # source_embeddings is size (num_source_domain_descriptions, num_classes, emb_size)
        if source_embeddings.shape[0] > 1:
            self.text_embeddings = target_embeddings - source_embeddings.mean(axis=0)
        else:
            self.text_embeddings = target_embeddings - source_embeddings

    # @staticmethod
    # def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    #     return [template.format(text) for template in templates]

    # @staticmethod
    # def get_embedding(text_prompts, model):
    #     text_inputs = torch.cat([clip.tokenize(t) for t in text_prompts]).cuda()
    #     # Calculate features
    #     with torch.no_grad():
    #         text_features = model.encode_text(text_inputs)
    #     return text_features.cpu().numpy()
    
    @staticmethod
    def normalize(inputs):
        try:
            inputs /= np.linalg.norm(inputs, axis=-1, keepdims=True)
        except:
            print("NOMRALIZE ERROR ", inputs, np.linalg.norm(inputs, axis=-1, keepdims=True))
        return inputs

    def apply(self, inputs, labels=None):
        if self.cfg.METHOD.NORMALIZE:
            return self.normalize(inputs)
        return inputs

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
        return self.inputs[idx], self.labels[idx], self.groups[idx], self.domain_labels[idx]

class ClipMLP(Base):

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
            print("----------- EVALUATING PREVIOUS CHECKPOINT -----------")
            self.load_checkpoint(self.cfg.METHOD.MODEL.CHECKPOINT)
    
        self.optimizer = AdamW(self.net.parameters(), lr=self.cfg.METHOD.MODEL.LR, weight_decay=self.cfg.METHOD.MODEL.WEIGHT_DECAY)
        self.create_criterion() # get loss functions

        if not self.cfg.METHOD.MODEL.RESUME:
            self.best_acc, self.best_epoch = 0, 0
            for epoch in range(self.cfg.EXP.EPOCHS):
                self.train_val_loop(self.train_loader, epoch, phase="train")
                if self.cfg.EXP.CHECKPOINT_VAL:
                    self.train_val_loop(self.test_loader, epoch, phase="val")
            self.save_checkpoint(0.0, epoch, last=True)
            wandb.summary['best val balanced acc'] = self.best_acc
            wandb.summary['best epoch'] = self.best_epoch

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])
        print(f"...loaded checkpoint with acc {checkpoint['acc']}")

    def predict(self, inp):
        out = self.net(inp)
        conf, cls_pred = torch.max(self.m(out), dim=-1)
        return out, conf, cls_pred

    def train_val_loop(self, loader, epoch, phase="train"):
        """
        One epoch of train-val loop.
        Returns of dict of metrics to log
        """
        total_loss, cls_correct, total = 0,0,0
        if phase == "train":
            self.net.train()
        else:
            self.net.eval()
        total_loss, cls_correct, total = 0, 0, 0
        cls_true, cls_pred, cls_groups, dom_true = np.array([]), np.array([]), np.array([]), np.array([])
        with torch.set_grad_enabled(phase == 'train'):
            for i, (inp, cls_target, cls_group, dom_target) in enumerate(loader):
                inp, cls_target = inp.cuda().float(), cls_target.cuda().long()
                if phase == "train":
                    self.optimizer.zero_grad()
                out, conf, cls_predicted = self.predict(inp)
                cls_loss = self.class_criterion(out.float(), cls_target)
                loss = cls_loss 
                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

                total_loss += cls_loss.item()
                total += cls_target.size(0)
                cls_correct += cls_predicted.eq(cls_target).sum().item()

                cls_true = np.append(cls_true, cls_target.cpu().numpy())
                cls_pred = np.append(cls_pred, cls_predicted.cpu().numpy())
                cls_groups = np.append(cls_groups, cls_group.cpu().numpy())
                dom_true = np.append(dom_true, dom_target.cpu().numpy())
                progress_bar(i, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (total/(i+1), 100.*cls_correct/total, cls_correct, total))
        
        accuracy, balanced_acc, class_accuracy, group_accuracy =  evaluate(cls_pred, cls_true, cls_groups)

        wandb.log({f"{phase} loss": total_loss, f"{phase} cls acc": accuracy, f"{phase} balanced class acc": balanced_acc, 
                    f"{phase} class acc": class_accuracy, f"{phase} group acc": group_accuracy, f"best {phase} balanced acc": self.best_acc})
        if phase == 'val' and balanced_acc > self.best_acc:
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
            # make checkpoint  directory
            torch.save(state, f'./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth')
            wandb.save(f'./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth')

class ClipMLPZS(ClipMLP):
    """
    Linear probing the CLIP embeddings, initializing the linear layer
    weights with the CLIP text embeddings.
    """
    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        super().__init__(text_prompts, model, cfg, neutral_prompts)
        templates = getattr(helpers.text_templates, cfg.EXP.TEMPLATES)
        # texts = [template.format(classname) for template in templates]
        text_embs = zeroshot_classifier([[p.format(c) for p in templates] for c in self.class_names], model, model_type=self.cfg.EXP.IMAGE_FEATURES, cuda_device='1')
        self.class_text_embs = text_embs.float().cuda()
        print("class text embs", self.class_text_embs.shape)

    def create_model(self, inputs):
        B, W  = inputs.shape
        self.model_conf = OmegaConf.create({"in_dim": W, "h_dim": self.cfg.METHOD.MODEL.HIDDEN_DIM, "out_dim": self.train_dataset.num_classes, "num_classes": self.train_dataset.num_classes, "num_domains": self.train_dataset.num_domains, "num_layers": self.cfg.METHOD.MODEL.NUM_LAYERS})
        self.cfg = OmegaConf.merge(self.cfg, self.model_conf)
        net = MPLZS(self.cfg, self.class_text_embs)
        self.net = net.cuda()
        net = torch.nn.DataParallel(self.net)
        cudnn.benchmark = True
    