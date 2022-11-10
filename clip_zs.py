from cgi import test
import os
import clip
import torch
import open_clip

import numpy as np
from tqdm import tqdm
import wandb
import argparse

from sklearn.metrics import confusion_matrix
from utils import read_unknowns, nest_dict

import helpers.data_helpers as dh
import methods.clip_transformations as CLIPTransformations
import helpers.text_templates
from clip_utils import get_features
import ast

from omegaconf import OmegaConf
import omegaconf

parser = argparse.ArgumentParser(description='CLIP Advice')
parser.add_argument('--config', default='configs/Noop.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
# flags = parser.parse_args()
flags, unknown = parser.parse_known_args()

overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base      = OmegaConf.load('configs/base.yaml')
args      = OmegaConf.merge(base, cfg, overrides)
if len(unknown) > 0:
    config = nest_dict(read_unknowns(unknown))
    to_merge = OmegaConf.create(config)
    args = OmegaConf.merge(args, to_merge)
args.yaml = flags.config

assert args.EXP.ADVICE_METHOD == 'CLIPZS', "clip_zs.py only for CLIPZS baseline, use train.py or clip_advice.py"

if args.EXP.WANDB_SILENT:
    os.environ['WANDB_SILENT']="true"

def flatten_config(dic, running_key=None, flattened_dict={}):
    for key, value in dic.items():
        if running_key is None:
            running_key_temp = key
        else:
            running_key_temp = '{}.{}'.format(running_key, key)
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            flatten_config(value, running_key_temp)
        else:
            flattened_dict[running_key_temp] = value
    return flattened_dict

run = wandb.init(project=args.EXP.PROJ, group=args.EXP.ADVICE_METHOD, config=flatten_config(args), entity="clipinvariance")
wandb.save(flags.config)
wandb.run.log_code(".")

torch.manual_seed(args.EXP.SEED)
np.random.seed(args.EXP.SEED)

DATASET_NAME = args.DATA.DATASET

# load data
if args.DATA.LOAD_CACHED:
    if args.EXP.IMAGE_FEATURES == 'clip':
        model_name = args.EXP.CLIP_MODEL
    elif args.EXP.IMAGE_FEATURES == 'openclip':
        model_name = args.EXP.CLIP_MODEL
    else:
        model_name = args.EXP.IMAGE_FEATURES
    cache_file, dataset_classes, dataset_domains = dh.get_cache_file(DATASET_NAME, model_name, args.EXP.BIASED_VAL, args.EXP.IMAGE_FEATURES)
    assert os.path.exists(cache_file), f"{cache_file} does not exist. To compute embeddings, set DATA.LOAD_CACHED=True"
    data = torch.load(cache_file)
    train_features, train_labels, train_groups, train_domains = data['train_features'], data['train_labels'], data['train_groups'], data['train_domains']
    val_features, val_labels, val_groups, val_domains = data['val_features'], data['val_labels'], data['val_groups'], data['val_domains']
    test_features, test_labels, test_groups, test_domains = data['test_features'], data['test_labels'], data['test_groups'], data['test_domains']
    if args.DATA.DATASET != 'ColoredMNISTBinary':
        val_features, val_labels, val_groups, val_domains, val_filenames = data['val_features'][::2], data['val_labels'][::2], data['val_groups'][::2], data['val_domains'][::2], data['val_filenames'][::2]
        test_features, test_labels, test_groups, test_domains, test_filenames = np.concatenate((data['test_features'], data['val_features'][1::2])), np.concatenate((data['test_labels'], data['val_labels'][1::2])), np.concatenate((data['test_groups'], data['val_groups'][1::2])), np.concatenate((data['test_domains'], data['val_domains'][1::2])), np.concatenate((data['test_filenames'], data['val_filenames'][1::2]))
        
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FEATURES = args.EXP.IMAGE_FEATURES
if IMAGE_FEATURES == 'clip':
    model, preprocess = clip.load(args.EXP.CLIP_MODEL, device)
elif IMAGE_FEATURES == 'openclip':
    model, _, preprocess = open_clip.create_model_and_transforms(args.EXP.CLIP_MODEL, pretrained=args.EXP.CLIP_PRETRAINED_DATASET)
    model = model.to(device)
model.eval()

if args.DATA.LOAD_CACHED ==  False:
    trainset, valset, testset = dh.get_dataset(DATASET_NAME, preprocess, biased_val=args.EXP.BIASED_VAL)
    dataset_classes = dh.get_class(DATASET_NAME)
    dataset_domains = dh.get_domain(DATASET_NAME)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)
    train_features, train_labels, train_groups, train_domains, train_filenames = get_features(train_loader, model, device, model_type=args.EXP.IMAGE_FEATURES)
    val_features, val_labels, val_groups, val_domains, val_filenames = get_features(val_loader, model, device, model_type=args.EXP.IMAGE_FEATURES)
    test_features, test_labels, test_groups, test_domains, test_filenames = get_features(test_loader, model, device, model_type=args.EXP.IMAGE_FEATURES)
    data = {
        "train_features": train_features,
        "train_labels": train_labels,
        "train_groups": train_groups,
        "train_domains": train_domains,
        "train_filenames": train_filenames,
        "val_features": val_features,
        "val_labels": val_labels,
        "val_groups": val_groups,
        "val_domains": val_domains,
        "val_filenames": val_filenames,
        "test_features": test_features,
        "test_labels": test_labels,
        "test_groups": test_groups,
        "test_domains": test_domains,
        "test_filenames": test_filenames
    }
    data_dir = '/'.join(args.DATA.SAVE_PATH.split('/')[:-1])

    if not os.path.exists(data_dir): os.makedirs(data_dir)

    torch.save(data, args.DATA.SAVE_PATH)

testset = CLIPTransformations.EmbeddingDataset(args, test_features, test_labels, test_groups, test_domains)
test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)

valset = CLIPTransformations.EmbeddingDataset(args, val_features, val_labels, val_groups, val_domains)
val_loader = torch.utils.data.DataLoader(valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)

@torch.no_grad()
def zeroshot_classifier(classnames, templates):
    model.to(device)
    model.eval()
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            if IMAGE_FEATURES == 'clip':
                texts = clip.tokenize(texts).cuda() #tokenize
            elif IMAGE_FEATURES == 'openclip':
                texts = open_clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights
print("TEMPLATES", getattr(helpers.text_templates, args.EXP.TEMPLATES))
zeroshot_weights = zeroshot_classifier(dataset_classes, getattr(helpers.text_templates, args.EXP.TEMPLATES))

def eval(loader):
    with torch.no_grad():
        image_feats, image_labels = [], []
        preds, labels, groups = np.array([]), np.array([]), np.array([])
        for i, (images, target, group, domains) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()
            images /= images.norm(dim=-1, keepdim=True)
            # predict
            logits = (100. * images @ zeroshot_weights).float().softmax(dim=-1)
            clip_pred = torch.argmax(logits, dim=-1)
            preds = np.append(preds, clip_pred.cpu().numpy())
            labels = np.append(labels, target.cpu().numpy())
            groups = np.append(groups, group.numpy())
            image_feats += [i for i in images]
            image_labels += [i for i in target]
    return preds, labels, groups


#validation set
preds, labels, groups = eval(val_loader)
accuracy, balanced_acc, class_accuracy, group_accuracy = CLIPTransformations.evaluate(preds, labels, groups)
# group_acc = group_accuracy.reshape(len(dataset_classes), max([int(len(group_accuracy)/len(dataset_classes)), 1]))
_, _, _, domain_accuracy = CLIPTransformations.evaluate(preds, labels, np.squeeze(val_domains), list(range(len(dataset_classes))))

wandb.summary["val acc"] = accuracy
wandb.summary["best val balanced acc"] = balanced_acc
wandb.summary["val class acc"] = class_accuracy

#test set
preds, labels, groups = eval(test_loader)
accuracy, balanced_acc, class_accuracy, group_accuracy = CLIPTransformations.evaluate(preds, labels, groups)
# group_acc = group_accuracy.reshape(len(dataset_classes), max([int(len(group_accuracy)/len(dataset_classes)), 1]))
_, _, _, domain_accuracy = CLIPTransformations.evaluate(preds, test_labels, np.squeeze(test_domains), list(range(len(dataset_classes))))
print(f"unique test domains {np.unique(test_domains)}")
wandb.summary["test acc"] = accuracy
wandb.summary["test blanced acc"] = balanced_acc
wandb.summary["test class acc"] = class_accuracy
wandb.summary["test domain acc"] = domain_accuracy
wandb.summary["test worst domain acc"] = np.min(domain_accuracy)
wandb.summary['test group acc'] = group_accuracy
for i in range(len(domain_accuracy)):
            wandb.summary[f"{dataset_domains[i]} test acc"] = domain_accuracy[i]
for i in range(len(dataset_classes)):
            wandb.summary[f"{dataset_classes[i]} test acc"] = class_accuracy[i]