import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# load Mask2Former fine-tuned on Cityscapes semantic segmentation
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-semantic"
)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-semantic"
)


def create_colormap(n):
    # Creates a colormap for n classes using the Pylab (matplotlib) color maps
    return plt.cm.get_cmap("hsv", n)


# use the function to create a colormap
colormap = create_colormap(20)


def segment_image(image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    print("class_queries_logits", class_queries_logits.shape)
    print("masks_queries_logits", masks_queries_logits.shape)

    # you can pass them to processor for postprocessing
    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    # convert tensor to numpy array
    semantic_map_array = predicted_semantic_map.squeeze().numpy()

    # normalize the indices to [0, 1] for the colormap
    semantic_map_array_normalized = semantic_map_array / (
        semantic_map_array.max() + 1e-9
    )

    # Apply color map to the semantic map
    semantic_map_color = (colormap(semantic_map_array_normalized) * 255).astype(
        np.uint8
    )

    # save the semantic map image
    semantic_map_image = Image.fromarray(semantic_map_color)
    semantic_map_image.save("semantic_map.png")

    # Overlay the semantic map on the original image
    original_image_array = np.array(image)
    original_image = Image.fromarray(original_image_array.astype(np.uint8))
    overlaid_image = Image.blend(
        original_image.convert("RGBA"), semantic_map_image.convert("RGBA"), alpha=0.5
    )
    overlaid_image.save("overlaid_image.png")
    os.remove("semantic_map.png")

    return predicted_semantic_map
