import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math
import torch.utils.model_zoo as model_zoo
import clip
import torchvision
import torch.nn.functional as F

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
    
class MPLZS(MLP):
    """
    MLP initialized with CLIP text embeddings
    """
    def __init__(self, cfg, text_embeddings):
        super(MPLZS, self).__init__(cfg)
        assert self.num_layers == 1, 'Only one layer supported'
        assert self.fc.weight.shape == text_embeddings.T.shape, f'Embedding dimension mismatch {self.fc.weight.shape} != {text_embeddings.shape}'
        self.fc.weight.data = nn.Parameter(text_embeddings.T.float().cuda())
        self.fc.weight.requires_grad = True
        nn.init.constant_(self.fc.bias.data, 0)
        self.fc.bias.requires_grad = True

class ResMLP(nn.Module):
    """
    We didn't end up using this because it didn't seem to help, but it's here if you want to try it.
    """
    def __init__(self, input_dim=768, hidden_dim=384, num_layers=1):
        super(ResMLP, self).__init__()
        self.num_layers = num_layers
        assert self.num_layers in [1,2,3], 'Only one or two # layers supported'
        if self.num_layers == 1:
            self.fc = nn.Linear(input_dim, input_dim)
        elif self.num_layers == 2:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, input_dim)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, input_dim)
            self.fc3 = nn.Linear(hidden_dim, input_dim)

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
    """
    Finetune the CLIP backbone. This theoretically should be usable....
    """
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
