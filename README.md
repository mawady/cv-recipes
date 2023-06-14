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
- ObjectClassificationBinary-FineTune-ResNet-CatsVsDogs-Pytorch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pRrX-2D5QOoLG4uzC58YFMm8wtCwmaJk?usp=sharing)
- ObjectClassificationBinary-Infer-ResNet-CatsVsDogs-Pytorch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cobRaHYrubud9-8BgjEnce-8957d5_5y?usp=sharing) [[Open In HuggingFace - Spaces](https://huggingface.co/spaces/mawady/demo-catsvsdogs-gradio)]
### Object detection
- ObjectDetection-Infer-YoloV3-COCO-OpenCV [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PKRX7TYwZ80YcvGYB-BjzA0QaqScwZJ1?usp=sharing)
### Object segmentation
- 
---

## Notes
- For single-label classification, mirco F1 and accruacy are equal. For multi-label, they are not.
- For imbalance dataset, use weighted F1 instead of macro F1.
- Training hacks/Fastai: progressive resizing, gradual unfreezing
- Baseline - Object Classification [single/multi class, single/multi label]: ResNet, ViT, EfficientNet
- Baseline - Object Detection [single/multi class, single/multi label]: YoloV3-5, RetinaNet, FasterRCNN
- Baseline - Object Segementation [semantic / instance]: MaskRCNN, U-NET, FCN, DeepLabV3
- MultilabelStratifiedKFold: [https://github.com/trent-b/iterative-stratification](https://github.com/trent-b/iterative-stratification)
