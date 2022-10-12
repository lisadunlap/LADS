# LADS
Official Implementation of LADS (Latent Augmentation using Domain descriptionS)

![LADS method overview.](figs/lads-method-2-1.png "LADS method overview")


## TODO

- [ ] finish readme
- [ ] check everything works (I would be surprised if anything runs rn)
- [ ] upload clip emb to gdrive to download
- [ ] upload checkpoints

## Getting started

1. Clone this repo (or if you are working with me fork this repo)

2. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yaml
    ```
3. Launch your environment with `conda activate LADS` or `source activate LADS`

4. Fix any misc bugs that you find :p

## Code Structure 
The configurations for each method are in the `configs` folder. To try say the baseline of doing normal LR on the CLIP embeddings:
```
python clip_advice.py --config configs/Waterbirds/Noop.yaml
```

Datasets supported are in the [helpers folder](./helpers/data_helpers.py). Currently they are:
* Waterbirds (100% and 95%)
* ColoredMNIST (LNTL version and simplified version)
* DomainNet
* CUB Paintings

You can download the CLIP embeddings of these datasets [here](https://drive.google.com/drive/folders/1ItjhX7RPfQ6fQQk6_bEYJPewnkVdcfOC?usp=sharing)

Since computing the CLIP embeddings for each train/val/test set is time consuming, you can store the embeddings by setting `DATA.LOAD_CACHED=False` and `DATA.SAVE_PATH=[path you want to save to]`

Then, add the path to the saved embeddings to DATASET_PATHS in [data_helpers](./helpers/data_helpers.py) and set `DATA.LOAD_CACHED=Tue` in your yaml file

More description of each method and the config files in the config folder. 

## Some important parameters
**EXP.TEXT_PROMPTS**

This is the domains/biases that you want to be invariant to. You can either have them be class specific (e.g. `["a painting of a {}.", "clipart of a {}."]`) or generic (e.g. `[["painting"], ["clipart"]]`). The default is class specific so if you want to use generic prompts instead set `AUGMENTATION.GENERIC=True`. For generic prompts, if you want to average the text embeddings of several phrases of a domain, simply add them to the list (e.g. `[["painting", "a photo of a painting", "an image of a painting"], ["clipart", "clipart of an object"]]`).

**EXP.NEUTRAL_PROMPTS**

If you want to take the difference in text embeddings (for things like the directional loss, most of the augmentations, and the embedding debiasing methods). you can set a neutral prompt (e.g. `["a sketch of a {}."]` or `[["a photo of a sketch]]`). Like TEXT_PROMPTS you can have it be class specific or generic, but if TEXT_PROMPTS is class specific so is NEUTRAL_PROMPTS and vice versa.

**EXP.ADVICE_METHOD**

This sets the type of linear probing you are doing. Set to `LR` if you want to use the scikit learn LR (what is in the CLIP repo) or `ClipMLP` for pytorch MLP (if `METHOD.MODEL.NUM_LAYERS=1` this is LR). Typically `CLIPMLP` runs a lot faster than `LR`.

You can also set the advice method to one of the debiasing methods (different from augmentations in that we augment the training data and dont add in the original training data), but we don't use them anymore and I'm too lazy to explain it so if you care to try them out check the configs file (WARNING these are old so high chance of bugs).

## Running CLIP Zero-Shot
In order to run the CLIP zero-shot baseline, set `EXP.ADVICE_METHOD=CLIPZS` and run the `clip_zs.py` file instead of `clip_advice.py` file. 

For example
```
python clip_zs.py --config configs/Waterbirds/ZS.yaml
```

CLIP text templates are located in `helpers/text_templates.py`, and you can specify which template you want with the `EXP.TEMPLATES` parameter. 

Also note that we use the classes given in `EXP.PROMPTS` instead of the dataset classes in the dataset object itself so make sure to set those correctly.

## Running LR

If you want to simply run logistic regression on the embeddings, run the `mlp.yaml` file in any of the config folders. Some of the methods we have dont require any training (e.g. HardDebias), so all those do is perform a transformation on the embeddings before we do the logistic regression. 

Note: you do need to save the embeddings for each model in the `helpers/dataset_helpers.py` folder.

For example, to run LR on CLIP with a resnet50 backbone on ColoredMNIST, run
```
python clip_advice.py --config configs/ColoredMNIST/mlp.yaml
```

## Directional Loss
The directional loss is an augmentation (so augment the training data and add it back in to the original). Its parameters are under the `AUGMENTATION` section of the config files. 

To make sure things are running correctly, run
`python clip_advice.py --config configs/DomainNetMini/DirectionalAll.yaml`
and check your results with https://wandb.ai/clipinvariance/CLIPInvariance_DomainNetMini/runs/1ctddl4j. 

## More examples
Here are some more examples for each method:

To train DANN on Waterbirds95:
``` 
python clip_advice.py --config configs/Waterbirds95/mlp.yaml DATA.DATASET=Waterbirds95 DATA.LOAD_CACHED=True MODEL.DOM_WEIGHT=0.001 MODEL.LR=0.001
```

To try [HardDebias](https://arxiv.org/pdf/1607.06520.pdf):
```
python clip_advice.py --config configs/ColoredMNIST/HardDebias.yaml
```

## Some Small things

In order to reuse code, some of the arguments for MLPDebias are required in the yaml file for MLP (e.g. `METHOD.USE_DOM_GT`). You will get an error if you dont include these arguments in your yaml file but you can set them however you want if you are training the MLP baseline, they wont be used. 
