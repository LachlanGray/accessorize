import os
import replicate
import cv2
import numpy as np

from cog import BasePredictor, Path, Input


def describe(prompt, image_path):
    image = open(image_path, "rb")
    output = replicate.run(
          "yorickvp/llava-13b:a0fdc44e4f2e1f20f2bb4e27846899953ac8e66c5886c5878fa1d6b73ce009e5",
          input={
              "image": image,
              "prompt": prompt
        }
    )
    return "".join(output)


def add_accessory(image_path, subject, accessory):
    image = open(image_path, "rb")

    output = replicate.run(
    "stability-ai/stable-diffusion-3.5-large",
        input={
            "cfg": 4.5,
            "image": image,
            "steps": 40,
            "prompt": f"{subject} wearing {accessory}",
            "aspect_ratio": "1:1",
            "output_format": "png",
            "output_quality": 90,
            "prompt_strength": 0.7
        }
    )

    save_path = f"{subject}_{accessory}.png"
    with open(save_path, 'wb') as f:
        f.write(output[0].read())

    return save_path


def get_mask(image_path, positive, negative):
    image = open(image_path, "rb")

    output = replicate.run(
        "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c",
        input={
            "image": image,
            "mask_prompt": positive,
            "adjustment_factor": 0,
            "negative_mask_prompt": "ground" # dummy
        }
    )

    mask = [x for x in output][2]

    # Save the generated image
    save_path = f"{positive}.png"
    with open(save_path, 'wb') as f:
        f.write(mask.read())

    return save_path


def combine_images(original_image_path, mask_image_path, second_image_path, output_image_path, blur_kernel_size=21):
    original_image = cv2.imread(original_image_path)
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    second_image = cv2.imread(second_image_path)

    if original_image.shape != second_image.shape:
        second_image = cv2.resize(second_image, (original_image.shape[1], original_image.shape[0]))
    if mask.shape != original_image.shape[:2]:
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

    smooth_mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

    alpha = smooth_mask / 255.0

    blended_image = (alpha[:, :, None] * second_image + (1 - alpha[:, :, None]) * original_image).astype(np.uint8)

    cv2.imwrite(output_image_path, blended_image)


class Predictor(BasePredictor):
    def predict(self,
            image: Path = Input(description="Image of unadorned subject"),
            accessory: str = Input(description="(optional) Accessory for subject; if empty one is chosen automatically", default=None)
    ) -> Path:

        subject = describe("In one word, what's the subject of this image?", image)
        print(f"subject: {subject}")

        if accessory is None:
            accessory = describe(f"In one word, what would be a fitting accessory for the {subject}?", image)

        print(f"accessory: {accessory}")

        accessorized = add_accessory(image, subject, accessory)
        print(f"accessorized image: {accessorized}")
        acc_mask = get_mask(accessorized, accessory, subject)
        print(f"mask {accessorized}")

        out_path = f"{subject}_{accessory}_final.png"
        combine_images(image, acc_mask, accessorized, out_path)
        print(f"final image: {out_path}")

        # clean up 
        os.remove(accessorized)
        os.remove(acc_mask)
        print("cleaned up")

        return out_path



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image of unadorned subject")
    parser.add_argument("--accessory", type=str, help="(optional) Accessory for subject; if empty one is chosen automatically", default=None)

    args = parser.parse_args()
    image = args.image
    accessory = args.accessory

    # image = Path("test_img/lion.png")
    # accessory = "Sunglasses"

    predictor = Predictor()

    output_path = predictor.predict(image=image, accessory=accessory)

