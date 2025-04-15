# overlay_utils.py

import cv2
import numpy as np
import torch
import torch.nn.functional as F

def img_save_and_viz(image, result, output_path, classes, palette, title=None,
                     opacity=0.5, threshold=0.3, texture=None):
    """
    Saves and visualizes the segmentation result by overlaying either a texture image
    on designated segmented regions (e.g., 'shirt') or applying a solid color for others.
    
    Parameters:
        image (torch.Tensor): The original BGR image as a Torch tensor.
        result (torch.Tensor): The segmentation result logits from the model.
        output_path (str): Path to save the output visualization.
        classes (list): List of class names.
        palette (list or dict): Mapping of class indices to BGR colors.
        title (str, optional): Title for the image (unused in current implementation).
        opacity (float): Blending factor for overlay (0-1).
        threshold (float): Threshold for converting logits in single-class prediction.
        texture (numpy.ndarray, optional): Texture image to overlay on the shirt region.
                                           This is applied when a class matches "shirt" (case-insensitive).
    """
    # Define output file names for the mask and segmentation array
    output_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", ".npy")
    )
    output_seg_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", "_seg.npy")
    )

    # Convert the original image (Torch tensor) to a numpy array (BGR)
    image = image.data.numpy()

    # Resize segmentation logits to the image dimensions
    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)

    # Generate prediction from logits
    if seg_logits.shape[0] > 1:
        pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True)
    else:
        seg_logits = seg_logits.sigmoid()
        pred_sem_seg = (seg_logits > threshold).to(seg_logits)

    pred_sem_seg = pred_sem_seg.data[0].numpy()
    print("Unique segmentation labels:", np.unique(pred_sem_seg))

    # Save the mask and segmentation array
    mask = pred_sem_seg > 0
    np.save(output_file, mask)
    np.save(output_seg_file, pred_sem_seg)

    num_classes = len(classes)
    sem_seg = pred_sem_seg
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    # Prepare an empty overlay image
    overlay = np.zeros_like(image)

    # If a texture is provided, resize it to match the original image dimensions
    if texture is not None:
        texture_resized = cv2.resize(texture, (image.shape[1], image.shape[0]))

    # Loop over each label and apply either a texture or a solid color from the palette
    # Loop over each label and apply either a texture or a solid color from the palette
    for label in labels:
        if texture is not None and label == 0:
            # Apply texture overlay for label 26 ("Torso")
            shirt_mask = (sem_seg == label)
            overlay[shirt_mask] = texture_resized[shirt_mask]
        else:
            color = palette[label]
            overlay[sem_seg == label, :] = color

    # Blend the original image and overlay based on opacity
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_image = (image_rgb * (1 - opacity) + overlay * opacity).astype(np.uint8)
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    # Optionally, concatenate the original image and the visualization
    vis_image = np.concatenate([image, vis_image], axis=1)
    cv2.imwrite(output_path, vis_image)