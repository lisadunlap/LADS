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
from clip_utils import *
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
# assert args.EXP.ADVICE_METHOD != 'CLIPZS', "main.py not for CLIP zero-shot, use clip_zs.py"

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

run = wandb.init(project=args.EXP.PROJ, group=args.EXP.ADVICE_METHOD, config=flatten_config(args), allow_val_change=False)
# wandb.save(flags.config)
# wandb.run.log_code(".")

torch.manual_seed(args.EXP.SEED)
np.random.seed(args.EXP.SEED)
random.seed(args.EXP.SEED)

DATASET_NAME = args.DATA.DATASET

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# # load data
# if args.DATA.LOAD_CACHED:
#     cache_file = f"{args.DATA.SAVE_PATH}/{args.DATA.DATASET}/{args.EXP.IMAGE_FEATURES}_{args.EXP.CLIP_PRETRAINED_DATASET}_{args.EXP.CLIP_MODEL.replace('/','_')}.pt"
#     dataset_classes, dataset_domains = dh.DATASET_CLASSES[args.DATA.DATASET], dh.DATASET_DOMAINS[args.DATA.DATASET]
#     assert os.path.exists(cache_file), f"{cache_file} does not exist. To compute embeddings, set DATA.LOAD_CACHED=False"
#     print(f"Loading cached embeddings from {cache_file}")
#     train_features, train_labels, train_groups, train_domains, train_filenames, val_features, val_labels, val_groups, val_domains, val_filenames, test_features, test_labels, test_groups, test_domains, test_filenames = load_embeddings(cache_file, args.DATA.DATASET)
cache_file = f"{args.DATA.SAVE_PATH}/{args.DATA.DATASET}/{args.EXP.IMAGE_FEATURES}_{args.EXP.CLIP_PRETRAINED_DATASET}_{args.EXP.CLIP_MODEL.replace('/','_')}.pt"
dataset_classes, dataset_domains = dh.DATASET_CLASSES[args.DATA.DATASET], dh.DATASET_DOMAINS[args.DATA.DATASET]
if os.path.exists(cache_file):
    print(f"Loading cached embeddings from {cache_file}")
    train_features, train_labels, train_groups, train_domains, train_filenames, val_features, val_labels, val_groups, val_domains, val_filenames, test_features, test_labels, test_groups, test_domains, test_filenames = load_embeddings(cache_file, args.DATA.DATASET)
else:
    print(f"Computing embeddings and saving to {cache_file}")
    trainset, valset, testset = dh.get_dataset(DATASET_NAME, preprocess, biased_val=args.EXP.BIASED_VAL)
    dataset_classes, dataset_domains = dh.get_class(DATASET_NAME), dh.get_domain(DATASET_NAME)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)
    train_features, train_labels, train_groups, train_domains, train_filenames = get_features(train_loader, model, device, model_type=args.EXP.IMAGE_FEATURES)
    val_features, val_labels, val_groups, val_domains, val_filenames = get_features(val_loader, model, device, model_type=args.EXP.IMAGE_FEATURES)
    test_features, test_labels, test_groups, test_domains, test_filenames = get_features(test_loader, model, device, model_type=args.EXP.IMAGE_FEATURES)
    save_dict = {
        "train_features": train_features, "train_labels": train_labels, "train_groups": train_groups, "train_domains": train_domains, "train_filenames": train_filenames,
        "val_features": val_features, "val_labels": val_labels, "val_groups": val_groups, "val_domains": val_domains, "val_filenames": val_filenames,
        "test_features": test_features, "test_labels": test_labels, "test_groups": test_groups, "test_domains": test_domains, "test_filenames": test_filenames,
        "seed": args.EXP.SEED
    }
    if not os.path.exists(f"{args.DATA.SAVE_PATH}/{args.DATA.DATASET}"):
        os.makedirs(f"{args.DATA.SAVE_PATH}/{args.DATA.DATASET}")
    cache_file = f"{args.DATA.SAVE_PATH}/{args.DATA.DATASET}/{args.EXP.IMAGE_FEATURES}_{args.EXP.CLIP_PRETRAINED_DATASET}_{args.EXP.CLIP_MODEL.replace('/','_')}.pt"
    torch.save(save_dict, cache_file)
    print(f"Saved CLIP embeddings to {cache_file}")

if args.METHOD.NORMALIZE:
    train_features /= np.linalg.norm(train_features, axis=-1, keepdims=True)
    val_features /= np.linalg.norm(val_features, axis=-1, keepdims=True)
    test_features /= np.linalg.norm(test_features, axis=-1, keepdims=True)

# Calculate the image features
prompts = list(args.EXP.TEXT_PROMPTS)
if len(prompts) >0 and type(prompts[0]) == omegaconf.listconfig.ListConfig:
    prompts = [list(p) for p in prompts]

neutral_prompts = list(args.EXP.NEUTRAL_TEXT_PROMPTS)
if len(neutral_prompts) >0 and type(neutral_prompts[0]) == omegaconf.listconfig.ListConfig:
    neutral_prompts = [list(p) for p in neutral_prompts]
print("Advice Method", args.EXP.ADVICE_METHOD)
bias_correction = getattr(CLIPTransformations, args.EXP.ADVICE_METHOD)(prompts, clip_model, args, neutral_prompts)

# old_train_features, old_train_labels, old_train_groups, old_train_domains, old_train_filenames = train_features, train_labels, train_groups, train_domains, train_filenames
old_val_features, old_val_labels, old_val_groups, old_val_domains, old_val_filenames = val_features, val_labels, val_groups, val_domains, val_filenames
old_test_features, old_test_labels, old_test_groups, old_test_domains, old_test_filenames = test_features, test_labels, test_groups, test_domains, test_filenames

# set zeroshot weights if doing a ensamble
if args.EXP.ENSAMBLE:
    all_prompts = neutral_prompts + prompts
    print("Setting zeroshot weights...")
    zeroshot_weights = zeroshot_classifier([[p.format(c) for p in all_prompts] for c in dataset_classes], model, model_type=args.EXP.IMAGE_FEATURES)
    dataset_doms = [d.replace('real', 'photo') for d in dataset_domains]
    dom_zeroshot_weights = zeroshot_classifier([[f"a {d} of an object."] for d in dataset_doms], model, model_type=args.EXP.IMAGE_FEATURES)
    print("Zeroshot weights set!")

# if we want to do any augmentations, do them here
num_augmentations = 1
if args.EXP.AUGMENTATION != None and args.EXP.AUGMENTATION != 'None':
    print("Augmenting training set...")
    if "LADS" in args.EXP.AUGMENTATION or 'Directional' in args.EXP.AUGMENTATION:
        augment = getattr(methods.augmentations, args.EXP.AUGMENTATION)(args, train_features, train_labels, train_groups, train_domains, train_filenames, bias_correction.text_embeddings, val_features, val_labels, val_groups, val_domains)
    else:
        augment = getattr(methods.augmentations, args.EXP.AUGMENTATION)(args, train_features, train_labels, train_groups, train_domains, train_filenames, bias_correction.text_embeddings)
    train_features, train_labels, train_domains, train_groups, train_filenames = augment.augment_dataset()
    print("Training set augmented!")

if args.EXP.LOG_NN:
        features, labels, groups, domains, filenames = np.concatenate([old_val_features, old_test_features]), np.concatenate([old_val_labels, old_test_labels]), np.concatenate([old_val_groups, old_test_groups]), np.concatenate([old_val_domains, old_test_domains]), np.concatenate([old_val_filenames, old_test_filenames])
        if len(np.unique(train_domains)) > 1:
            filtered_idxs = np.where(train_domains != train_domains[0])
            sample_features, sample_domains, sample_labels, sample_filenames = np.array(train_features[filtered_idxs]), train_domains[filtered_idxs], train_labels[filtered_idxs], train_filenames[filtered_idxs]
            sample_idxs = random.sample(list(range(len(sample_filenames))), min((len(train_filenames), 1000)))
            sample_features, sample_domains, sample_labels, sample_filenames = sample_features[sample_idxs], sample_domains[sample_idxs], sample_labels[sample_idxs], sample_filenames[sample_idxs]
        else:
            sample_idxs = random.sample(list(range(len(train_filenames))), min((len(train_filenames), 1000)))
            sample_features, sample_domains, sample_labels, sample_filenames = train_features[sample_idxs], train_domains[sample_idxs], train_labels[sample_idxs], train_filenames[sample_idxs]
        neighbor_domains, neighbor_labels, domain_acc, class_acc, neighbor_samples, prop_unique, mean_cs = get_nn_metrics(sample_features, sample_domains, sample_labels, features, domains, labels)
        plt.rcParams["figure.figsize"] = (20,5)
        f, (axs_orig, axs_new) = plt.subplots(2, 10, sharey=True)
        for i, (original_idx, sample_idx) in enumerate(neighbor_samples):
            try:
                axs_orig[i].imshow(Image.open(sample_filenames[original_idx]).resize((224, 224)))
                axs_orig[i].set_title(f"{dataset_domains[int(sample_domains[int(original_idx)])]} - {sample_labels[int(original_idx)]}")
                axs_orig[i].axis('off')
                axs_new[i].imshow(Image.open(filenames[sample_idx]).resize((224, 224)))
                axs_new[i].set_title(f"{dataset_domains[int(domains[int(sample_idx)])]} - {labels[int(sample_idx)]}")
                axs_new[i].axis('off')
            except:
                print(f"sample idx {sample_idx} is not a valid index")
        wandb.log({"train features NN": wandb.Image(f), "domain consistency acc": domain_acc, "class consistency acc": class_acc, "unique nn": prop_unique})
        # wandb.sklearn.plot_confusion_matrix(sample_domains, neighbor_domains, dataset_domains)
        print("Plotted Nearest Neighbors")

# train MLP with domain adaptation loss
bias_correction.train_debias(train_features, train_labels, train_groups, train_domains, val_features, val_labels, np.squeeze(val_groups), val_domains)
if args.EXP.ENSAMBLE:
    print("Ensambling predictions")
    predictions, probs = bias_correction.eval(val_features)
    lads_preds, zs_preds, ensamble_predictions, combined_preds = get_ensamble_preds(val_features, probs, zeroshot_weights, dataset_domains=dom_zeroshot_weights)
    non_overlap, non_overlap_prop, non_overlap_prop_correct = get_pred_overlap(lads_preds, zs_preds, val_labels)
    accuracy, balanced_acc, class_accuracy, group_accuracy = evaluate(ensamble_predictions, val_labels, np.squeeze(val_groups), num_augmentations=num_augmentations)
    wandb.summary["ensamble val acc"] = accuracy
    wandb.summary["ensamble val blanced acc"] = balanced_acc

    predictions, probs = bias_correction.eval(test_features)
    lads_preds, zs_preds, ensamble_predictions, combined_preds = get_ensamble_preds(test_features, probs, zeroshot_weights, dataset_domains=dom_zeroshot_weights)
    non_overlap, non_overlap_prop, non_overlap_prop_correct = get_pred_overlap(lads_preds, zs_preds, test_labels)
    accuracy, balanced_acc, class_accuracy, group_accuracy = evaluate(ensamble_predictions, test_labels, np.squeeze(test_groups), num_augmentations=num_augmentations)
    _, _, _, domain_accuracy = evaluate(ensamble_predictions, test_labels, np.squeeze(test_domains), list(range(len(dataset_classes))), num_augmentations=num_augmentations)
    wandb.summary["ensamble test acc"] = accuracy
    wandb.summary["ensamble test blanced acc"] = balanced_acc
    wandb.summary["ensamble test class acc"] = class_accuracy
    wandb.summary["ensamble test domain acc"] = domain_accuracy
    wandb.summary["ensamble test worst domain acc"] = np.min(domain_accuracy)
    wandb.summary['ensamble test group acc'] = group_accuracy   
else:
    predictions, probs = bias_correction.eval(test_features)
val_predictions, val_probs = bias_correction.eval(val_features)
val_accuracy, val_balanced_acc, val_class_accuracy, val_group_accuracy = evaluate(val_predictions, val_labels, np.squeeze(val_groups), num_augmentations=num_augmentations)
accuracy, balanced_acc, class_accuracy, group_accuracy = evaluate(predictions, test_labels, np.squeeze(test_groups), num_augmentations=num_augmentations)
_, _, _, domain_accuracy = evaluate(predictions, test_labels, np.squeeze(test_domains), list(range(len(dataset_classes))), num_augmentations=num_augmentations)
wandb.summary["test acc"] = accuracy
wandb.summary["test blanced acc"] = balanced_acc
wandb.summary["test class acc"] = class_accuracy
wandb.summary["test domain acc"] = domain_accuracy
wandb.summary["test worst domain acc"] = np.min(domain_accuracy)
wandb.summary['test group acc'] = group_accuracy
for i in range(len(domain_accuracy)):
    wandb.summary[f"{dataset_domains[i]} test acc"] = domain_accuracy[i]
print(f"Test accuracy: {group_accuracy} \n Test domain accuracy: {domain_accuracy}")

if 'E2E' in args.EXP.ADVICE_METHOD:
    # features, labels, groups, domains, filenames = np.concatenate([old_val_features, old_test_features]), np.concatenate([old_val_labels, old_test_labels]), np.concatenate([old_val_groups, old_test_groups]), np.concatenate([old_val_domains, old_test_domains]), np.concatenate([old_val_filenames, old_test_filenames])
    aug_features, aug_labels, aug_domains, aug_filenames = bias_correction.augment_dataset(train_features, train_labels, train_domains, train_filenames)
    sample_idxs = random.sample(list(range(len(aug_filenames))), 1000)
    # print("SAMPLE SHAPE: ", sample_filenames.shape, sample_domains.shape)
    sample_features, sample_domains, sample_labels, sample_filenames = aug_features[sample_idxs], aug_domains[sample_idxs], aug_labels[sample_idxs], aug_filenames[sample_idxs]
    neighbor_domains, neighbor_labels, domain_acc, class_acc, neighbor_samples, prop_unique, mean_cs = get_nn_metrics(sample_features, sample_domains, sample_labels, old_test_features, old_test_domains, old_test_labels)
    wandb.log({"mean CS for NN": mean_cs})
    print(neighbor_samples)
    plt.rcParams["figure.figsize"] = (20,5)
    f, (axs_orig, axs_new) = plt.subplots(2, 10, sharey=True)
    for i, (original_idx, sample_idx) in enumerate(neighbor_samples):
        # try:
        print(sample_filenames[original_idx])
        axs_orig[i].imshow(Image.open(sample_filenames[original_idx]).resize((224, 224)))
        axs_orig[i].set_title(f"{dataset_domains[int(sample_domains[int(original_idx)])]} - {sample_labels[int(original_idx)]}")
        axs_orig[i].axis('off')
        axs_new[i].imshow(Image.open(old_test_filenames[sample_idx]).resize((224, 224)))
        axs_new[i].set_title(f"{dataset_domains[int(old_test_domains[int(sample_idx)])]} - {old_test_labels[int(sample_idx)]}")
        axs_new[i].axis('off')
        # except:
        #     print(f"sample idx {sample_idx} is not a valid index")
    wandb.log({"train features NN": wandb.Image(f), "domain consistency acc": domain_acc, "class consistency acc": class_acc, "unique nn": prop_unique})
    wandb.sklearn.plot_confusion_matrix(sample_domains, neighbor_domains, dataset_domains)