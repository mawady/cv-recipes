# Recipes for computer vision - educational and research use

---

## Filenaming: Task-Stage-Model-Dataset-Framework
- Task: ObjectClassification, ObjectDetection, etc...
- Stage: Train, FineTune, Eval, Infer
- Model: ResNet, YoloV3, etc...
- Dataset: ImageNet, COCO, etc...
- Library: PyTorch, Tensorflow, Keras

---

## Basic level
### Object classification
- ObjectClassificationBinary_FineTune_ResNet_CatsVsDogs_Pytorch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mawady/colab-recipes-cv/blob/main/ObjectClassificationBinary_FineTune_ResNet_CatsVsDogs_Pytorch.ipynb)
- ObjectClassificationBinary_Infer_ResNet_CatsVsDogs_Pytorch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mawady/colab-recipes-cv/blob/main/ObjectClassificationBinary_Infer_ResNet_CatsVsDogs_Pytorch.ipynb) [![Open In HuggingFace - Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/mawady/demo-catsvsdogs-gradio)
- ObjectClassificationMultiClass_FineTune_ResNet_MNIST_Pytorch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mawady/cv-recipes/blob/main/ObjectClassificationMultiClass_FineTune_ResNet_MNIST_Pytorch.ipynb)
### Object detection
- ObjectDetection_Infer_YoloV3_COCO_OpenCV [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mawady/colab-recipes-cv/blob/main/ObjectDetection_Infer_YoloV3_COCO_OpenCV.ipynb)
### Object segmentation
- ObjectSegmentationPanoptic_Infer_DETR_COCO_PyTorch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mawady/cv-recipes/blob/main/ObjectSegmentationPanoptic_Infer_DETR_COCO_PyTorch.ipynb)
- ObjectSegmentationInstance_Infer_MaskRCNN_COCO_PyTorch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mawady/cv-recipes/blob/main/ObjectSegmentationInstance_Infer_MaskRCNN_COCO_PyTorch.ipynb)
---

## Notes
- For single-label classification, mirco F1 and accruacy are equal. For multi-label, they are not.
- For imbalance dataset, use weighted F1 instead of macro F1.
- Training hacks/Fastai: progressive resizing, gradual unfreezing
- Baseline - Object Classification [single/multi class, single/multi label]: ResNet, ViT, EfficientNet
- Baseline - Object Detection [single/multi class, single/multi label]: YoloV3-5, RetinaNet, FasterRCNN
- Baseline - Object Segementation [semantic / instance]: MaskRCNN, U-NET, FCN, DeepLabV3
- MultilabelStratifiedKFold: [https://github.com/trent-b/iterative-stratification](https://github.com/trent-b/iterative-stratification)
