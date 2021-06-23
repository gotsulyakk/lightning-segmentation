import cv2
import yaml
import torch
import utils
import argparse
import numpy as np
from typing import Dict
from skimage import color
from skimage import segmentation
import segmentation_models_pytorch as smp


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image", type=str, help="Path to the image.", required=True)
    parser.add_argument("-m", "--model_ckpt", type=str, help="Path to the model ckpt", required=True)
    parser.add_argument("-c", "--config", type=str, help="Path to the config", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(args.config) as f:
            hparams = yaml.load(f, Loader=yaml.SafeLoader)

    model = utils.object_from_dict(hparams["model"])

    corrections: Dict[str, str] = {"model.": ""}

    state_dict = utils.state_dict_from_disk(
                    file_path=args.model_ckpt,
                    rename_in_layers=corrections,
                )
    model.load_state_dict(state_dict)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
                hparams["model"]["encoder_name"], 
                hparams["model"]["encoder_weights"]
    )
    transform = utils.get_validation_aug(preprocessing_fn)
    image_data = transform(image=image)

    model.eval()
    with torch.no_grad():
        result = model(image_data["image"].unsqueeze(0))

    result = (result.squeeze().cpu().numpy().round())

    segmentation_result = cv2.resize(result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    label2rgb = color.label2rgb(segmentation_result, image)
    img_with_contours = segmentation.mark_boundaries(image, segmentation_result, mode='thick')
    img_with_mask = cv2.addWeighted(
        image, 1, (cv2.cvtColor(segmentation_result, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0
    )

    utils.visualize(
        original_image=image,
        predicted_mask=segmentation_result,
        label2rgb=label2rgb,
        image_with_contour=img_with_contours,
        image_with_mask=img_with_mask
    )


if __name__ == "__main__":
    main()