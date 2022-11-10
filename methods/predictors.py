import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math
import torch.utils.model_zoo as model_zoo
import clip
import torchvision

from pytorch_revgrad import RevGrad
from omegaconf import OmegaConf
import torch.nn.functional as F

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

class MPLZS(MLP):
    def __init__(self, cfg, text_embeddings):
        super(MPLZS, self).__init__(cfg)
        assert self.num_layers == 1, 'Only one layer supported'
        assert self.fc.weight.shape == text_embeddings.T.shape, f'Embedding dimension mismatch {self.fc.weight.shape} != {text_embeddings.shape}'
        self.fc.weight.data = nn.Parameter(text_embeddings.T.float().cuda())
        self.fc.weight.requires_grad = True
        nn.init.constant_(self.fc.bias.data, 0)
        self.fc.bias.requires_grad = True

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

class DPLMLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width=512, dropout=0.0, mlp_depth=3):
        super(DPLMLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width,mlp_width)
            for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class DPLCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        nn.Module.__init__(self)
        n_cls = len(classnames)
        self.n_ctx = cfg.TRAINER.COCOOP.N_CTX
        self.avg = cfg.METHOD.BATCH_AVERAGING
        self.ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        self.ctx_load = cfg.TRAINER.COCOOP.LOAD_CTX
        n_dom = cfg.TRAINER.COCOOP.NUM_DOM_TOKEN
        self.classnames = classnames
        self.n_dom = n_dom
        self.ctx_checkpoint = cfg.TRAINER.COCOOP.CTX_CHECKPOINT
        self.clip_model = clip_model.cuda()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        vis_dim = clip_model.visual.output_dim
        self.avg_test = cfg.METHOD.TEST_BATCH_AVG
        clip_imsize = clip_model.visual.input_resolution
        print("INPUT ", cfg.INPUT)
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.setup(n_dom)

    def setup(self, n_dom):

        #  initial prompt.
        prompt_prefix = ' '.join(['X'] * n_dom)
        
        if self.ctx_init:
            print('Using sentence_prompt in DPLCLIP...')
            classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.classnames]
        else:
            classnames = [name.replace('_', ' ') for name in self.classnames]
        prompts = [prompt_prefix + ' ' + name + '.' for name in self.classnames]
        # prompts:  ['X X X X X X X X dog.', 'X X X X X X X X elephant.' ...]
        
        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)
        
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        
        self.register_buffer('token_suffix', embedding[:, n_dom + 1:, :])  # CLS, EOS
        # torch.Size([7, 68, self.EMBEDDING_DIM]), 68 := 77 - num_domain_tokens_tokens - 2.
        # [ 0.0013,  0.0046, -0.0115,  ...,  0.0112,  0.0147,  0.0040],...,.
        
        # self.network = networks.MLP(self.EMBEDDING_DIM, self.EMBEDDING_DIM * n_dom, hparams).to(device=self.device, dtype=self.clip_model.dtype)
        # self.network = torchvision.ops.MLP(self.ctx_dim, [self.ctx_dim, self.ctx_dim, self.ctx_dim * n_dom], dropout=0.1).cuda().to(self.clip_model.dtype)
        self.network = DPLMLP(self.ctx_dim, self.ctx_dim * n_dom, dropout=0.1).cuda().half()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=5e-5,
            momentum=0.1
        )
        
        self.network.apply(init_weights)
        print(self.network)
        
            
    def update(self, image_features, labels, unlabeled=None):
        # minibatches = [[domain_1, labels_1], [domain_2], [domain_3]]
        # image_features = [data[0].cuda().half() for data in minibatches]
        # all_y = [data[1].cuda().long() for data in minibatches]
        image_features, all_y = [image_features], labels.long()

        #  encode image for each domain.
        # image_features = [self.clip_model.encode_image(x) for x in all_x]
        
        #  extract domain_feature for each domain. [32, self.EMBEDDING_DIM] -> [32, self.EMBEDDING_DIM * num_domain_tokens] -> [self.EMBEDDING_DIM * num_domain_tokens].
        domain_features = [self.network(feature) for feature in image_features]
        image_features = torch.cat(image_features)
        if self.avg:
        #  reshape [self.batch_size, self.EMBEDDING_DIM.]:  -> [1, self.EMBEDDING_DIM.]
            mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]
            #  reshape [1, self.EMBEDDING_DIM.]:  -> [7, self.EMBEDDING_DIM.]
            _mean_domain_features = [feature.repeat_interleave(len(self.classnames), dim=0) for feature in mean_domain_features]
            #  generate text_feature from domain_feature. text_features.size = [3, 7, 512]
            # text_features = [self._get_text_features(feature) for feature in _mean_domain_features]
            text_features = torch.cat([self._get_text_features(feature) for feature in _mean_domain_features])
                
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_per_image = self.clip_model.logit_scale.exp() * image_features @ text_features.t()
            
        else:
            # REMOVE MEAN STEPS
            domain_features = torch.cat(domain_features)
            domain_features = [feature[None, :] for feature in domain_features]
            
            #  reshape [1, self.EMBEDDING_DIM.]:  -> [7, self.EMBEDDING_DIM.]
            #mean_domain_features = 
            _new_domain_features = [feature.repeat_interleave(len(self.classnames), dim=0) for feature in domain_features]
            
            #  generate text_feature from domain_feature. text_features.size = [3, 7, 512]
            #text_features = torch.cat([self._get_text_features(feature) for feature in _new_domain_features])
            text_features = torch.stack([self._get_text_features(feature) for feature in _new_domain_features], dim=1)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            #logits_per_image = self.clip_model.logit_scale.exp() * image_features @ text_features.t()

            logits_per_image = (self.clip_model.logit_scale.exp() * torch.matmul(text_features, image_features.t()).sum(1)).t()
        
        loss = F.cross_entropy(logits_per_image, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return logits_per_image, loss


    def _get_text_features(self, domain_feature, coop=False):
        #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
        dom_feat_before = domain_feature
        domain_feature = domain_feature.reshape(-1, self.n_dom, self.ctx_dim)
        
        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        domain_feature = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)
        
        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        x = domain_feature + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        
        #  mapping domain_features to text_features.
        text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection      
        return text_features

    def predict(self, image_feature):
        # image_feature = self.clip_model.encode_image(x)
        domain_feature = self.network(image_feature)

        if self.avg_test:
            mean_domain_feature = torch.mean(domain_feature, dim=0, keepdim=True).repeat_interleave(len(self.classnames), dim=0)
            text_feature = self._get_text_features(mean_domain_feature)
            
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            logits = self.clip_model.logit_scale.exp() * image_feature @ text_feature.t()
        else:
            new_domain_feature = [feature.repeat_interleave(len(self.classnames), dim=0) for feature in domain_feature]
            text_feature = [self._get_text_features(feat) for feat in new_domain_feature]
            #text_feature = self._get_text_features(mean_domain_feature)
            text_feature = torch.stack(text_feature, dim=1)
            
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            #text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            
            logits = (self.clip_model.logit_scale.exp() * torch.matmul(text_feature, image_feature.t()).sum(1)).t()
            
        conf, preds = torch.max(logits.softmax(dim=-1), dim=-1)
        return logits, conf, preds

# from methods.predictors import DPLCLIP

# class DPL(CoCoOp):
#     """
#     Domain Prompt Learner
#     """
#     def __init__(self, cfg, clip_model, zeroshot_weights, dataset_classes, dataset_domains):
#         super().__init__(cfg, clip_model, zeroshot_weights, dataset_classes, dataset_domains)
#         self.clip_model = clip_model.cpu()
#         self.clip_model.eval()
#         self.log_scale = clip_model.logit_scale
#         self.prompt_learner = DPLCLIP(cfg, dataset_classes, self.clip_model)
#         self.prompt_learner = self.prompt_learner.cuda()
#         self.best_acc, self.best_path = 0, None
#         for name, param in self.clip_model.named_parameters():
#             param.requires_grad_(False)

#         enabled = set()
#         for name, param in self.prompt_learner.named_parameters():
#             if param.requires_grad:
#                 enabled.add(name)
        
#         wandb.watch(self.prompt_learner, log="all", log_freq=10)
#         print(f"Parameters to be updated: {enabled}")
        
#     def save_model(self, model, epoch, acc, tag=None):
#         state = {
#             "acc": acc,
#             "epoch": epoch,
#             "net": model.network.state_dict()
#         }
#         if tag:
#             save_path = f'./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-{self.uid}-{tag}.pth'
#         else:
#             save_path = f'./checkpoint/{self.cfg.DATA.DATASET}/{self.cfg.METHOD.MODEL.CHECKPOINT_NAME}-{self.cfg.EXP.SEED}-epoch{epoch}-{self.uid}.pth'
#         save_dir = f'./checkpoint/{self.cfg.DATA.DATASET}'
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         print(f'Saving checkpoint with acc {acc} to {save_path}...')
#         torch.save(state, save_path)
#         wandb.save(save_path)
#         return save_path

#     def load_model(self, path=None):
#         if not os.path.exists(path):
#             raise ValueError(f"checkpoint {path} does not exist!")
#         checkpoint = torch.load(path)
#         self.prompt_learner.network.load_state_dict(checkpoint['net'])
#         print(f"...loaded checkpoint with acc {checkpoint['acc']}...")

#     def train_on_source(self, train_loader, val_loader):
#         self.start_epoch = 0
#         if self.cfg.METHOD.MODEL.RESUME or self.cfg.METHOD.MODEL.EVAL:
#             print(f"resume from {self.cfg.METHOD.MODEL.RESUME_CHECKPOINT}")
#             self.best_path = self.cfg.METHOD.MODEL.RESUME_CHECKPOINT
#             checkpoint = torch.load(self.cfg.METHOD.MODEL.RESUME_CHECKPOINT)
#             self.start_epoch, self.best_acc = checkpoint['epoch'], checkpoint['acc']
#         if not self.cfg.METHOD.MODEL.EVAL:
#             for epoch in range(self.start_epoch, self.cfg.METHOD.SOURCE_EPOCHS):
#                 train_results = self.train_val_loop(train_loader, phase="train")
#                 val_results = self.train_val_loop(val_loader, phase="val")
#                 train_results.update(val_results)
#                 train_results.update({"epoch": epoch, "lr": self.prompt_learner.optimizer.param_groups[0]["lr"]})
#                 if train_results['val_accuracy'] > self.best_acc:
#                     self.best_acc = train_results['val_accuracy']
#                     save_path = self.save_model(self.prompt_learner, epoch, train_results['val_accuracy'], tag='best')
#                     self.best_path = save_path
#                 wandb.log(train_results)
#                 if epoch % self.cfg.METHOD.MODEL.SAVE_EVERY == 0:
#                     save_path = self.save_model(self.prompt_learner, epoch, train_results['val_accuracy'], tag=None)
#         wandb.summary['best val acc'] = self.best_acc
#         self.load_model(self.best_path)
#         self.prompt_learner.eval()


#     def train_val_loop(self, loader, phase="train"):
#         """
#         One epoch of train-val loop.
#         Returns of dict of metrics to log
#         """
#         total_loss, cls_correct, total = 0,0,0
#         if phase == "train":
#             self.prompt_learner.train()
#         else:
#             self.prompt_learner.eval()
#         with torch.set_grad_enabled(phase == 'train'):
#             for i, (inp, cls_target, cls_group, dom_target) in enumerate(loader):
#                 inp, cls_target = inp.cuda(), cls_target.cuda()
#                 if phase == "train":
#                     logits, loss = self.prompt_learner.update(inp, cls_target)
#                     conf, preds = torch.max(logits.softmax(dim=-1), dim=-1)
#                 else:
#                     logits, conf, preds = self.prompt_learner.predict(inp)
#                     loss = F.cross_entropy(logits, cls_target.long())

#                 total_loss += loss.item()
#                 total += cls_target.size(0)
#                 cls_correct += preds.eq(cls_target).sum().item()
#                 wandb.log({f"{phase} step loss": total_loss/(i+1), f"{phase} acc": 100.*cls_correct/total})
#                 progress_bar(i, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                             % (total_loss/(i+1), 100.*cls_correct/total, cls_correct, total))

#         return {f"{phase}_loss": total_loss/(i+1), f"{phase}_accuracy": 100.*cls_correct/total}

#     def predict(self, img_embeddings, label=None):
#         return self.prompt_learner.predict(img_embeddings.cuda())