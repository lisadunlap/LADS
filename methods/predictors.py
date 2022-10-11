import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math
import torch.utils.model_zoo as model_zoo
import clip
import torchvision

from pytorch_revgrad import RevGrad
from omegaconf import OmegaConf

class Predictor(nn.Module):
    def __init__(self, input_ch=32, num_classes=8):
        super(Predictor, self).__init__()
        self.pred_conv1 = nn.Conv2d(input_ch, input_ch, kernel_size=3,
                                    stride=1, padding=1)
        self.pred_bn1   = nn.BatchNorm2d(input_ch)
        self.relu       = nn.ReLU(inplace=True)
        self.pred_conv2 = nn.Conv2d(input_ch, num_classes, kernel_size=3,
                                    stride=1, padding=1)
        self.softmax    = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pred_conv1(x)
        x = self.pred_bn1(x)
        x = self.relu(x)
        x = self.pred_conv2(x)
        px = self.softmax(x)

        return x,px

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.num_layers = cfg["num_layers"]
        assert self.num_layers in [1,2,3], 'Only one or two # layers supported'
        if self.num_layers == 1:
            self.fc = nn.Linear(cfg["in_dim"], cfg["out_dim"])
        elif self.num_layers == 2:
            self.fc1 = nn.Linear(cfg["in_dim"], cfg["h_dim"])
            self.fc2 = nn.Linear(cfg["h_dim"], cfg["out_dim"])
        else:
            self.fc1 = nn.Linear(cfg["in_dim"], cfg["h_dim"])
            self.fc2 = nn.Linear(cfg["h_dim"], cfg["h_dim"])
            self.fc3 = nn.Linear(cfg["h_dim"], cfg["out_dim"])

    def forward(self, x):
        if self.num_layers == 1:
            h = self.fc(x)
        elif self.num_layers == 2:
            h = nnf.relu(self.fc1(x))
            h = self.fc2(h)
        else:
            h = nnf.relu(self.fc1(x))
            h = nnf.relu(self.fc2(h))
            h = self.fc3()
        return h

class ResMLP(nn.Module):
    def __init__(self, cfg):
        super(ResMLP, self).__init__()
        self.num_layers = cfg["num_layers"]
        assert self.num_layers in [1,2,3], 'Only one or two # layers supported'
        if self.num_layers == 1:
            self.fc = nn.Linear(cfg["in_dim"], cfg["out_dim"])
        elif self.num_layers == 2:
            self.fc1 = nn.Linear(cfg["in_dim"], cfg["h_dim"])
            self.fc2 = nn.Linear(cfg["h_dim"], cfg["out_dim"])
        else:
            self.fc1 = nn.Linear(cfg["in_dim"], cfg["h_dim"])
            self.fc2 = nn.Linear(cfg["h_dim"], cfg["h_dim"])
            self.fc3 = nn.Linear(cfg["h_dim"], cfg["out_dim"])

    def forward(self, x):
        if self.num_layers == 1:
            h = self.fc(x)
        elif self.num_layers == 2:
            h = nnf.relu(self.fc1(x))
            h = self.fc2(h)
        else:
            h = nnf.relu(self.fc1(x))
            h = nnf.relu(self.fc2(h))
            h = self.fc3(h)
        return x + h

class EmbeddingDebiasModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.mlp = MLP(cfg)
        self.cfg = cfg
        if not cfg.METHOD.MODEL.SEPERATE_CLASSES: assert cfg["num_classes"]==2, "SEPERATE CLASSES = False only when its a binary classification"
        num_outputs = 1 if not cfg.METHOD.MODEL.SEPERATE_CLASSES else cfg["num_classes"]
        cls_model_conf = OmegaConf.create({"in_dim": cfg["out_dim"], "out_dim": num_outputs, "num_layers": cfg.METHOD.MODEL.NUM_CLASS_LAYERS})
        cls_cfg = OmegaConf.merge(cfg, cls_model_conf)
        self.classifier_head = MLP(cls_cfg)
        # self.domain_head = nn.Sequential(RevGrad(alpha=cfg.METHOD.MODEL.DOM_WEIGHT), nn.Linear(cfg["out_dim"], cfg['num_domains']))
        self.domain_head = nn.Sequential()
        self.domain_head.add_module("lin", nn.Linear(cfg["out_dim"], cfg['num_domains']))

    def forward(self, x, alpha = 0.1):
        x = self.mlp(x)
        if self.cfg.METHOD.MODEL.WEAK_LABELS:
            d = self.domain_head(x)
        else:
            reverse_x = GradReverse.apply(x, alpha)     
            d = self.domain_head(reverse_x)
        c = self.classifier_head(x)
        return c, d

class DANN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.base_model = torchvision.models.resnet50(pretrained=True)
        self.cfg = cfg
        if not cfg.METHOD.MODEL.SEPERATE_CLASSES: assert cfg["num_classes"]==2, "SEPERATE CLASSES = False only when its a binary classification"
        num_outputs = 1 if not cfg.METHOD.MODEL.SEPERATE_CLASSES else cfg["num_classes"]
        cls_model_conf = OmegaConf.create({"in_dim": cfg["out_dim"], "out_dim": num_outputs, "num_layers": cfg.METHOD.MODEL.NUM_CLASS_LAYERS})
        cls_cfg = OmegaConf.merge(cfg, cls_model_conf)
        self.classifier_head = MLP(cls_cfg)
        # self.domain_head = nn.Sequential(RevGrad(alpha=cfg.METHOD.MODEL.DOM_WEIGHT), nn.Linear(cfg["out_dim"], cfg['num_domains']))
        self.domain_head = nn.Sequential()
        self.domain_head.add_module("lin", nn.Linear(cfg["out_dim"], cfg['num_domains']))

    def forward(self, x, alpha = 0.1):
        x = self.mlp(x)
        if self.cfg.METHOD.MODEL.WEAK_LABELS:
            d = self.domain_head(x)
        else:
            reverse_x = GradReverse.apply(x, alpha)     
            d = self.domain_head(reverse_x)
        c = self.classifier_head(x)
        return c, d

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # try increasing alpha
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x):
    return GradReverse.apply(x)

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp16)

class CLIPFinetune(nn.Module):

    def __init__(self, clip_model, num_classes=8):
        super(CLIPFinetune, self).__init__()
        convert_weights(clip_model)
        self.clip_model = clip_model
        self.fc = nn.Linear(clip_model.output_dim, num_classes)

    def forward(self, x):
        x = self.clip_model(x)
        x = self.fc(x)
        return x


class SALEME2E(nn.Module):

    def __init__(self, clip_model, config, num_classes=8):
        super(SALEME2E, self).__init__()
        self.backbone = clip_model
        self.augmentation_net = MLP(config['augmentation_config'])
        self.classifier = MLP(config['classifier_config'])

    def forward(self, x):
        img_emb = self.backbone(x)
        aug_img_emb = self.augmentation_net(x)
        out_real = self.classifier(img_emb)
        out_aug = self.classifier(aug_img_emb)
        return img_emb, aug_img_emb, out_real, out_aug