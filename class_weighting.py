import os
import clip
import open_clip
import torch
import numpy as np
import torchvision
import wandb
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import random
import omegaconf
from omegaconf import OmegaConf

import helpers.data_helpers as dh
import methods.clip_transformations as CLIPTransformations
from utils import read_unknowns, nest_dict
from clip_utils import get_features, evaluate, zeroshot_classifier, get_ensamble_preds, get_pred_overlap, get_nn_metrics
import methods.augmentations

parser = argparse.ArgumentParser(description='CLIP Advice')
parser.add_argument('--config', default='configs/base.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
# flags = parser.parse_args()
flags, unknown = parser.parse_known_args()

overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base      = OmegaConf.load('configs/base.yaml')
args      = OmegaConf.merge(base, cfg, overrides)
if len(unknown) > 0:
    print(unknown)
    config = nest_dict(read_unknowns(unknown))
    to_merge = OmegaConf.create(config)
    args = OmegaConf.merge(args, to_merge)
args.yaml = flags.config

assert args.EXP.ADVICE_METHOD != 'CNN', "main.py not for CNN baseline, use train.py"
assert args.EXP.ADVICE_METHOD != 'CLIPZS', "main.py not for CLIP zero-shot, use clip_zs.py"

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
            #print(running_key_temp, value)
            flattened_dict[running_key_temp] = value
    return flattened_dict

run = wandb.init(project='debug', group=args.EXP.ADVICE_METHOD, config=flatten_config(args), allow_val_change=False)
# wandb.save(flags.config)
# wandb.run.log_code(".")

torch.manual_seed(args.EXP.SEED)
np.random.seed(args.EXP.SEED)
random.seed(args.EXP.SEED)

DATASET_NAME = args.DATA.DATASET

# load data
if args.DATA.LOAD_CACHED:
    print(args.DATA.LOAD_CACHED)
    if args.EXP.IMAGE_FEATURES == 'clip' or args.EXP.IMAGE_FEATURES == 'openclip':
        model_name = args.EXP.CLIP_MODEL
    else:
        model_name = args.EXP.IMAGE_FEATURES
    cache_file, dataset_classes, dataset_domains = dh.get_cache_file(DATASET_NAME, model_name, args.EXP.BIASED_VAL, args.EXP.IMAGE_FEATURES)
    assert os.path.exists(cache_file), f"{cache_file} does not exist. To compute embeddings, set DATA.LOAD_CACHED=False"
    data = torch.load(cache_file)
    train_features, train_labels, train_groups, train_domains, train_filenames = data['train_features'], data['train_labels'], data['train_groups'], data['train_domains'], data['train_filenames']
    val_features, val_labels, val_groups, val_domains, val_filenames = data['val_features'], data['val_labels'], data['val_groups'], data['val_domains'], data['val_filenames']
    test_features, test_labels, test_groups, test_domains, test_filenames = data['test_features'], data['test_labels'], data['test_groups'], data['test_domains'], data['test_filenames']
    # move some val data to test 
    if args.DATA.DATASET != 'ColoredMNISTBinary':
        val_features, val_labels, val_groups, val_domains, val_filenames = data['val_features'][::2], data['val_labels'][::2], data['val_groups'][::2], data['val_domains'][::2], data['val_filenames'][::2]
        test_features, test_labels, test_groups, test_domains, test_filenames = np.concatenate((data['test_features'], data['val_features'][1::2])), np.concatenate((data['test_labels'], data['val_labels'][1::2])), np.concatenate((data['test_groups'], data['val_groups'][1::2])), np.concatenate((data['test_domains'], data['val_domains'][1::2])), np.concatenate((data['test_filenames'], data['val_filenames'][1::2]))
    if args.METHOD.NORMALIZE:
        train_features /= np.linalg.norm(train_features, axis=-1, keepdims=True)
        val_features /= np.linalg.norm(val_features, axis=-1, keepdims=True)
        test_features /= np.linalg.norm(test_features, axis=-1, keepdims=True)
# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.EXP.IMAGE_FEATURES)
# clip_model, preprocess = clip.load(args.EXP.CLIP_MODEL, device)
if args.EXP.IMAGE_FEATURES == 'clip':
    clip_model, preprocess = clip.load(args.EXP.CLIP_MODEL, device)
    model, preprocess = clip.load(args.EXP.CLIP_MODEL, device)
elif args.EXP.IMAGE_FEATURES == 'openclip':
    model, _, preprocess = open_clip.create_model_and_transforms(args.EXP.CLIP_MODEL, pretrained=args.EXP.CLIP_PRETRAINED_DATASET)
    model = model.to(torch.device('cuda:0'))
    clip_model = model
else:
    model = getattr(torchvision.models, args.EXP.IMAGE_FEATURES)(pretrained=True)
    model = model.to(device)

# Calculate the image features
prompts = list(args.EXP.TEXT_PROMPTS)
if len(prompts) >0 and type(prompts[0]) == omegaconf.listconfig.ListConfig:
    prompts = [list(p) for p in prompts]

neutral_prompts = list(args.EXP.NEUTRAL_TEXT_PROMPTS)
if len(neutral_prompts) >0 and type(neutral_prompts[0]) == omegaconf.listconfig.ListConfig:
    neutral_prompts = [list(p) for p in neutral_prompts]

lp = CLIPTransformations.ClipMLP(prompts, clip_model, args, neutral_prompts)
zs = CLIPTransformations.CLIPZS(prompts, clip_model, args, neutral_prompts)

# train MLP with domain adaptation loss
lp.train_debias(train_features, train_labels, train_groups, train_domains, val_features, val_labels, np.squeeze(val_groups), val_domains)

lp_val_predictions, lp_val_probs = lp.eval(val_features)
zs_val_predictions, zs_val_probs = zs.eval(val_features)

def log_wandb(acc, balanced_acc, class_acc, group_acc, tag='val'):
    wandb.summary[f"{tag}_accuracy"] = acc
    wandb.summary[f"{tag}_balanced_accuracy"] = balanced_acc
    wandb.summary[f"{tag}_group_accuracy"] = group_acc
    for d, d_acc in zip(dataset_domains, group_acc):
        wandb.summary[f"{tag}_{d}_acc"] = d_acc

lp_val_accuracy, lp_val_balanced_acc, lp_val_class_accuracy, lp_val_group_accuracy = evaluate(lp_val_predictions, val_labels, np.squeeze(val_groups))
zs_val_accuracy, zs_val_balanced_acc, zs_val_class_accuracy, zs_val_group_accuracy = evaluate(zs_val_predictions, val_labels, np.squeeze(val_groups))
log_wandb(lp_val_accuracy, lp_val_balanced_acc, lp_val_class_accuracy, lp_val_group_accuracy, tag='lp_val')
log_wandb(zs_val_accuracy, zs_val_balanced_acc, zs_val_class_accuracy, zs_val_group_accuracy, tag='zs_val')

print('..........................................')
print(f"LP val accuracy: {lp_val_accuracy} \t ZS val accuracy: {zs_val_accuracy}")
# acc_diff = lp_val_class_accuracy - zs_val_class_accuracy
# print(lp_val_class_accuracy + zs_val_class_accuracy)
acc_prop = np.nan_to_num(zs_val_class_accuracy / (lp_val_class_accuracy + zs_val_class_accuracy), nan=0.5)
print(f"--------------- acc prop {acc_prop} {type(acc_prop)} {acc_prop.shape} np sum {np.sum(acc_prop)}")
class_weights = acc_prop / np.sum(acc_prop)
print(f"Accuracy difference: {class_weights[:10]} \t Accuracy proportion: {acc_prop[:10]} {np.sum(acc_prop)}")
print('..........................................')

old_val_features, old_val_labels, old_val_groups, old_val_domains, old_val_filenames = val_features, val_labels, val_groups, val_domains, val_filenames
old_test_features, old_test_labels, old_test_groups, old_test_domains, old_test_filenames = test_features, test_labels, test_groups, test_domains, test_filenames

if args.EXP.AUGMENTATION != None and args.EXP.AUGMENTATION != 'None':
    print("Augmenting training set...")
    if "LADS" in args.EXP.AUGMENTATION or 'Directional' in args.EXP.AUGMENTATION:
        augment = getattr(methods.augmentations, args.EXP.AUGMENTATION)(args, train_features, train_labels, train_groups, train_domains, train_filenames, lp.text_embeddings, val_features, val_labels, val_groups, val_domains, class_weights)
    else:
        augment = getattr(methods.augmentations, args.EXP.AUGMENTATION)(args, train_features, train_labels, train_groups, train_domains, train_filenames, lp.text_embeddings)
    train_features, train_labels, train_domains, train_groups, train_filenames = augment.augment_dataset()
    print("Training set augmented!")

# if args.EXP.LOG_NN:
#         features, labels, groups, domains, filenames = np.concatenate([old_val_features, old_test_features]), np.concatenate([old_val_labels, old_test_labels]), np.concatenate([old_val_groups, old_test_groups]), np.concatenate([old_val_domains, old_test_domains]), np.concatenate([old_val_filenames, old_test_filenames])
#         # features, labels, groups, domains, filenames = old_test_features, old_test_labels, old_test_groups, old_test_domains, old_test_filenames
#         if len(np.unique(train_domains)) > 1:
#             filtered_idxs = np.where(train_domains != train_domains[0])
#             sample_features, sample_domains, sample_labels, sample_filenames = np.array(train_features[filtered_idxs]), train_domains[filtered_idxs], train_labels[filtered_idxs], train_filenames[filtered_idxs]
#             sample_idxs = random.sample(list(range(len(sample_filenames))), min((len(train_filenames), 1000)))
#             sample_features, sample_domains, sample_labels, sample_filenames = sample_features[sample_idxs], sample_domains[sample_idxs], sample_labels[sample_idxs], sample_filenames[sample_idxs]
#         else:
#             sample_idxs = random.sample(list(range(len(train_filenames))), min((len(train_filenames), 1000)))
#             sample_features, sample_domains, sample_labels, sample_filenames = train_features[sample_idxs], train_domains[sample_idxs], train_labels[sample_idxs], train_filenames[sample_idxs]
#         neighbor_domains, neighbor_labels, domain_acc, class_acc, neighbor_samples, prop_unique, mean_cs = get_nn_metrics(sample_features, sample_domains, sample_labels, features, domains, labels)
#         plt.rcParams["figure.figsize"] = (20,5)
#         f, (axs_orig, axs_new) = plt.subplots(2, 10, sharey=True)
#         for i, (original_idx, sample_idx) in enumerate(neighbor_samples):
#             try:
#                 axs_orig[i].imshow(Image.open(sample_filenames[original_idx]).resize((224, 224)))
#                 axs_orig[i].set_title(f"{dataset_domains[int(sample_domains[int(original_idx)])]} - {sample_labels[int(original_idx)]}")
#                 axs_orig[i].axis('off')
#                 axs_new[i].imshow(Image.open(filenames[sample_idx]).resize((224, 224)))
#                 axs_new[i].set_title(f"{dataset_domains[int(domains[int(sample_idx)])]} - {labels[int(sample_idx)]}")
#                 axs_new[i].axis('off')
#             except:
#                 print(f"sample idx {sample_idx} is not a valid index")
#         wandb.log({"train features NN": wandb.Image(f), "domain consistency acc": domain_acc, "class consistency acc": class_acc, "unique nn": prop_unique})
#         # wandb.sklearn.plot_confusion_matrix(sample_domains, neighbor_domains, dataset_domains)
#         print("Plotted Nearest Neighbors")

# retrain the model on the augmented dataset
lp.train_debias(train_features, train_labels, train_groups, train_domains, val_features, val_labels, np.squeeze(val_groups), val_domains)
lp_val_predictions, lp_val_probs = lp.eval(val_features)
lp_val_accuracy, lp_val_balanced_acc, lp_val_class_accuracy, lp_val_group_accuracy = evaluate(lp_val_predictions, val_labels, np.squeeze(val_groups))
log_wandb(lp_val_accuracy, lp_val_balanced_acc, lp_val_class_accuracy, lp_val_group_accuracy, tag='aug_lp_val')
lp_test_predictions, lp_test_probs = lp.eval(test_features)
lp_test_accuracy, lp_test_balanced_acc, lp_test_class_accuracy, lp_test_group_accuracy = evaluate(lp_test_predictions, test_labels, np.squeeze(test_groups))
log_wandb(lp_test_accuracy, lp_test_balanced_acc, lp_test_class_accuracy, lp_test_group_accuracy, tag='aug_lp_test')
