DATASET_PATHS = {
    "ViT-B/32": {
                "Waterbirds": 'data/waterbirds/wb_100_vit.pth',
                "Waterbirds95": 'data/waterbirds/clip_embeddings_95.pth',
                "Planes": 'data/planes/clip_embeddings_bias_A.pth',
                "ColoredMNIST": 'data/ColoredMNIST/clip_embeddings.pth',
                "CatsDogs": 'data/CatsDogs/bright_dogs_dark_cats.pth',
                "PlanesExt": 'data/planes/planes_ext_vit.pth',
                "MNIST": "data/ColoredMNIST/mnist.pth",
                "Imagenet": "data/Imagenet/emb.pth",
                "Imagenet-a": "data/Imagenet/imagenet_a.pth"
                },
}

BIASED_DATASET_PATHS = {
    "clip":{
        "ViT-B/32": {
                "Waterbirds": 'data/waterbirds/wb_100_vit_bval.pth',
                "Waterbirds95": 'data/waterbirds/wb_95_vit_bval.pth',
                "ColoredMNIST": 'data/ColoredMNIST/emb_vit_biased.pth',
                "Planes": "data/planes/emb_vit_bval.pth",
                "PlanesExt": "data/planes/emb_vit_ext_bval.pth",
                "ColoredMNISTBinary": "data/ColoredMNIST/binary.pth",
                "ColoredMNISTQuinque": "data/ColoredMNIST/quinque.pth",
                "MNIST": "data/ColoredMNIST/mnist.pth",
                "Imagenet": "data/Imagenet/emb.pth",
                "Imagenet-a": "data/Imagenet/imagenet_a.pth"
                },
        "RN50":     {
                "Waterbirds95": 'data/waterbirds/wb_95_rn50_bval.pth',
                "Waterbirds": 'data/waterbirds/rn50_clip.pth',
                "ColoredMNIST": 'data/ColoredMNIST/lntl_rn50.pth',
                "PlanesBalanced": "data/planes/balanced_rn50.pth",
                "Living17": "data/Living17/rn50_clip.pth",
                "DomainNetMini": 'data/DomainNetMini/rn50_clip.pth',
                "ColoredMNISTQuinque": "data/ColoredMNIST/rn50_clip_quinque.pth",
                "DomainNetMiniOracle": "data/DomainNetMini/rn50_oracle_clip.pth",
                "ColoredMNISTBinary": "data/ColoredMNIST/final_colored_mnist.pth",
                "DomainNetMiniAug": "data/DomainNetMini/aug_rn50_clip.pth",
                "CUB": "data/CUB/rn50_clip.pth",
                "SVHN": "data/SVHN/svhn_rn50.pth",
                "MNIST_SVHN": "data/SVHN/mnist_svhn_rn50.pth"
                },
        "ViT-L/14": {
                "CUB": "data/CUB/vit14_clip.pth",
                "DomainNetMini": "data/DomainNetMini/vit14_clip.pth",
                "MNIST_SVHN": "data/SVHN/vit14_clip.pth",
                "Waterbirds": "data/waterbirds/vit14_clip.pth",
                "ColoredMNISTBinary": "data/ColoredMNIST/binary_vit14.pth"
                },
        "resnet50": {
                "Waterbirds": 'data/waterbirds/resnet_waterbirds.pth',
        }
    },
    "openclip":{
        "ViT-H-14": {
                "ColoredMNISTBinary": "data/ColoredMNIST/colored_mnist_vith14.pth",
                },
        "ViT-g-14": {
                "ColoredMNISTBinary": "data/ColoredMNIST/colored_mnist_vitg14.pth",
                },
        "ViT-L-14": {
                "ColoredMNISTBinary": "data/ColoredMNIST/colored_mnist_vitL14.pth",
                },
    }
    

}