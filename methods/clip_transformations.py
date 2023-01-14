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
from clip_utils import zeroshot_classifier, evaluate
from methods.lads_utils import get_domain_text_embs, EmbeddingDataset, DirectionLoss

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

# class EmbeddingDataset:
#     """
#     Takes in CLIP embeddings (INPUTS), labels, and CLIP text embedding (TEXT_EMB of shape (num_domains, clip emb shape)).
#     Weakly labels the domain using the text embeddings 
#     TODO: try softlabels
#     """
#     def __init__(self, cfg, inputs, labels, groups, dom_gt, text_emb=[]):
#         self.inputs, self.labels, self.groups = inputs, labels, groups
#         self.text_emb = text_emb
#         self.cfg = cfg
#         _, self.embedding_dim = inputs.shape
#         if self.cfg.METHOD.USE_DOM_GT:
#             print("==> Using domain GT labels")
#             self.domain_labels = dom_gt
#         else:
#             print("==> Using domain CLIP labels")
#             self.domain_labels = self.get_labels(self.text_emb, self.inputs) if len(self.text_emb) > 0 else np.array([0 for i in range(len(self.inputs))])
#         self.num_classes, self.num_domains = len(set(self.labels)), len(set(self.domain_labels))
#         # get class weights for upweighting
#         self.class_weights = self.get_counts(self.labels)
#         self.dom_weights = self.get_counts(self.domain_labels)
#         assert len(self.inputs) == len(self.labels) == len(self.domain_labels), "input, label, and domain label lengths don't match"

#     @staticmethod
#     def get_counts(labels):
#         values, counts = np.unique(labels, return_counts=True)
#         sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
#         values, counts = [ list(tuple) for tuple in  sorted_tuples]
#         fracs   = 1 / torch.Tensor(counts)
#         return fracs / torch.max(fracs)

#     @staticmethod
#     def get_labels(text_emb, inputs):
#         """ Gets weak domain labels given CLIP text embeddings """
#         if len(text_emb.shape) == 3:
#             text_emb = torch.mean(text_emb, dim = 0)
#             print("new text emb ", text_emb.shape)
#         similarity = (100.0 * torch.Tensor(inputs).to(device).float() @ text_emb.T.to(device).float()).softmax(dim=-1)
#         values, indices = similarity.topk(1)
#         return [i[0].item() for i in indices]

#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, idx):
#         return self.inputs[idx], self.labels[idx], self.groups[idx], self.domain_labels[idx]

class ClipMLP(Base):

    def __init__(self, text_prompts, model, cfg, neutral_prompts=[]):
        super().__init__(text_prompts, model, cfg, neutral_prompts)
        self.text_emb = []
        self.cfg = cfg
        self.uid = uuid.uuid4()

    def get_checkpoint_name(self, last=False):
        if last:
            return f"./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}-last.pth"
        return f"./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth"

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
            # wandb.ss['best val balanced acc'] = self.best_acc
            # wandb.ss['best epoch'] = self.best_epoch

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
                progress_bar(i, len(self.test_loader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                            % (total/(i+1), 100.*cls_correct/total, cls_correct, total))
        
        accuracy, balanced_acc, class_accuracy, group_accuracy =  evaluate(cls_pred, cls_true, cls_groups)

        wandb.log({f"{phase} loss": total_loss, f"{phase} cls acc": accuracy, f"{phase} balanced class acc": balanced_acc, 
                    f"{phase} class acc": class_accuracy, f"{phase} group acc": group_accuracy, f"best {phase} balanced acc": self.best_acc})
        if phase == 'val' and balanced_acc > self.best_acc:
            self.best_acc, self.best_epoch = balanced_acc, epoch
            self.save_checkpoint(balanced_acc, epoch)

    def eval(self, inputs, ret_probs=False):
        """ Forward pass for classification. if probs=True, return the softmax prob (for ensambling) """

        ckpt = self.get_checkpoint_name()
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
            # wandb.save(f'./checkpoint/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}-last.pth')
        else:
            # make checkpoint  directory
            torch.save(state, f'./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth')
            # wandb.save(f'./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}.pth')

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
        print(f'Saving checkpoint with acc {acc} to .{self.get_checkpoint_name()}...')
        state = {
            "acc": acc,
            "epoch": epoch,
            "net": self.net.state_dict(),
            "aug_net": self.aug_net.state_dict()
        }
        if last:
            torch.save(state, f'{self.get_checkpoint_name()[:-4]}-last.pth')
            wandb.save(f'{self.get_checkpoint_name()[:-4]}-last.pth')
        else:
            torch.save(state, self.get_checkpoint_name())
            wandb.save(self.get_checkpoint_name())

    def create_model(self, inputs):
        B, W  = inputs.shape
        self.model_conf = OmegaConf.create({"in_dim": W, "h_dim": W, "out_dim": self.train_dataset.num_classes, "num_classes": self.train_dataset.num_classes, "num_domains": self.train_dataset.num_domains, "num_layers": self.cfg.METHOD.MODEL.NUM_LAYERS})
        self.cfg = OmegaConf.merge(self.cfg, self.model_conf)
        net = MLP(self.cfg)
        self.net = net.cuda()
        # net = torch.nn.DataParallel(self.net)
        cudnn.benchmark = True
        self.augmentation_model_conf = OmegaConf.create({"in_dim": W, "h_dim": self.cfg.AUGMENTATION.MODEL.HIDDEN_DIM, "out_dim": W, "num_classes": self.train_dataset.num_classes, "num_layers": self.cfg.AUGMENTATION.MODEL.NUM_LAYERS})
        self.num_domains = len(self.text_embeddings)
        aug_net = nn.ModuleList([MLP(OmegaConf.merge(self.cfg, self.augmentation_model_conf)) for i in range(self.num_domains)])
        self.aug_net = aug_net.cuda()
        # self.aug_net = torch.nn.DataParallel(self.aug_net)
        wandb.watch(aug_net, log_freq=10)
        print("NUM DOMAINS ", len(self.text_embeddings), self.train_dataset.num_domains)

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
            text_embs = zeroshot_classifier(prompts, self.model, model_type=self.cfg.EXP.IMAGE_FEATURES).cpu().numpy().T
            text_embs /= np.linalg.norm(text_embs, axis=-1, keepdims=True)
            return text_embs
        else:
            all_texts, text_names = [], []
            for t in prompts:
                texts = [[t.format(c)] for c in self.class_names]
                text_names += texts
                text_emb = zeroshot_classifier(texts, self.model, model_type=self.cfg.EXP.IMAGE_FEATURES).cpu().numpy().T
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
                num_domains, num_classes, emb_dim = self.text_embeddings.shape
                text_emb = torch.Tensor(self.text_embeddings[domain]).reshape((num_classes, 1, emb_dim)).cuda()

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
            self.load_checkpoint(self.get_checkpoint_name())
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
                    num_domains, num_classes, emb_dim = self.text_embeddings.shape
                    text_emb = torch.Tensor(self.text_embeddings[domain]).reshape((num_classes, 1, emb_dim)).cuda()

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
        ckpt = self.get_checkpoint_name()
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