# Domain Style Perturbation [Unofficial]

> Reference: Kill Two Birds with One Stone: Domain Generalization for Semantic Segmentation via Network Pruning

![](.assets/DSP.jpg)

## 1. Dataset Organization

Content Dataset: COCO ; Style Dataset : WikiArt.

---

Here is the training result:

![](.assets/training_result.jpg)

## 2. Ckpt of VGG

Considering the simplicity of the code, I extracted the checkpoints for the first 31 layers of VGG.

And you can download it from [the release page](https://github.com/Kiumb1223/DomainStylePerturbation-Unofficial/releases/tag/vgg-ckpt) --- the file is named `encoder_first_31_layers_vgg.pth`. 

Please place it under `checkpoints/` directory.

## 3. Environment 

Here are a few of the more important package versions I’ve listed; the list may not be complete.

```bash
cudatoolkit==11.3.1

pytorch==1.12.1
torchvision==0.13.1
tensorboard==2.20.0

numpy==1.26.2
opencv-python==4.9.0.80
hydra-core==1.3.2
pillow==9.5.0
loguru
easydict 
tabulate

```

## 4. Training

Before start to train Domain Style Perturbation, I recommend you to check [configuration file](config/config.yaml) first.

Then,
```bash
python train.py
```

## 5. Inference

```bash
# set path to the content image and path to output directory 
python inference.py -c path/to/content_img -o path/to/output_dir
```

## Reference

   1. [Kill Two Birds with One Stone: Domain Generalization for Semantic Segmentation via Network Pruning](https://arxiv.org/abs/2504.21019)
   2. [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)