import numpy as np
from matplotlib import pyplot as plt

def convert_yolo_to_pixel(coords, img_width, img_height):
    x_center, y_center, width, height = coords
    x_center_pixel = x_center * img_width
    y_center_pixel = y_center * img_height
    width_pixel = width * img_width
    height_pixel = height * img_height
    
    # Compute the top-left corner and the bottom-right corner
    x0 = x_center_pixel - width_pixel / 2
    y0 = y_center_pixel - height_pixel / 2
    x1 = x_center_pixel + width_pixel / 2
    y1 = y_center_pixel + height_pixel / 2

    return x0, y0, x1, y1

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))
