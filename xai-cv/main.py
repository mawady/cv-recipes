import warnings

warnings.filterwarnings("ignore")
import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import urllib.request
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import gradio as gr

IMG_SIZE = 224
CLASSES = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
TOP_NUM_CLASSES = 3

url = (
    "https://upload.wikimedia.org/wikipedia/commons/3/38/Adorable-animal-cat-20787.jpg"
)
path_input = "./cat.jpg"
urllib.request.urlretrieve(url, filename=path_input)


url = "https://upload.wikimedia.org/wikipedia/commons/4/43/Cute_dog.jpg"
path_input = "./dog.jpg"
urllib.request.urlretrieve(url, filename=path_input)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model = resnet18(pretrained=True)

data_transforms = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def grad_campp(img, cls_ids):
    img_rz = cv2.resize(np.array(img), (IMG_SIZE, IMG_SIZE))
    img = np.float32(img_rz) / 255
    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(
        device
    )
    # mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

    # Set target layers
    target_layers = [model.layer4[-1]]

    # Set target classes
    # targets = [ClassifierOutputTarget(cls_id) for cls_id in cls_ids]

    # GradCAM++
    gradcampp = GradCAMPlusPlus(model=model, target_layers=target_layers)

    lst_gradcam = []
    for i in range(TOP_NUM_CLASSES):
        targets = [ClassifierOutputTarget(cls_ids[i])]
        grayscale_gradcampp = gradcampp(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=False,
            aug_smooth=False,
        )
        grayscale_gradcampp = grayscale_gradcampp[0, :]
        gradcampp_image = show_cam_on_image(img, grayscale_gradcampp, use_rgb=True)
        lst_gradcam.append(gradcampp_image)

    return img_rz, lst_gradcam


def do_inference(img):
    img_t = data_transforms(img)
    batch_t = torch.unsqueeze(img_t, 0)
    model.eval()
    # We don't need gradients for test, so wrap in
    # no_grad to save memory
    with torch.no_grad():
        batch_t = batch_t.to(device)
        # forward propagation
        output = model(batch_t)
        # get prediction
        probs = torch.nn.functional.softmax(output, dim=1)
        cls_ids = (
            torch.argsort(probs, dim=1, descending=True).cpu().numpy()[0].astype(int)
        )[:TOP_NUM_CLASSES]
        probs = probs.cpu().numpy()[0]
        probs = probs[cls_ids]
        labels = np.array(CLASSES)[cls_ids]
    img_rz, lst_gradcam = grad_campp(img, cls_ids)
    return (
        {labels[i]: round(float(probs[i]), 2) for i in range(len(labels))},
        img_rz,
        lst_gradcam[0],
        lst_gradcam[1],
        lst_gradcam[2],
    )


im = gr.inputs.Image(
    shape=None, image_mode="RGB", invert_colors=False, source="upload", type="pil"
)

title = "Explainable AI - PyTorch"
description = "Playground: GradCam Inferernce of Object Classification using ResNet18 model. Libraries: PyTorch, Gradio, Grad-Cam"
examples = [["./cat.jpg"], ["./dog.jpg"]]
article = "<p style='text-align: center'><a href='https://github.com/mawady' target='_blank'>By Dr. Mohamed Elawady</a></p>"
iface = gr.Interface(
    do_inference,
    im,
    outputs=[
        gr.outputs.Label(num_top_classes=TOP_NUM_CLASSES),
        gr.outputs.Image(label="Output image", type="pil"),
        gr.outputs.Image(label="Output image", type="pil"),
        gr.outputs.Image(label="Output image", type="pil"),
        gr.outputs.Image(label="Output image", type="pil"),
    ],
    live=False,
    interpretation=None,
    title=title,
    description=description,
    examples=examples,
)

# iface.test_launch()

iface.launch()
