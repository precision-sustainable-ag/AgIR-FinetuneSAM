import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from omegaconf import DictConfig


class ModelComparator:
    def __init__(self, model_type, checkpoint, device='cuda'):
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device
        self.sam_model_orig = None
        self.predictor_tuned = None
        self.predictor_original = None
        self.image = None
    
    def load_models(self):
        self.sam_model_orig = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam_model_orig.to(self.device)
  
    def setup_predictors(self, cfg:DictConfig, bbox_coords):
        self.predictor_tuned = SamPredictor(self.sam_model)
        self.predictor_original = SamPredictor(self.sam_model_orig)
        k = 'NCA03593'
        self.image = cv2.imread(f'{cfg.data.images}/{k}.jpg')
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.predictor_tuned.set_image(self.image)
        self.predictor_original.set_image(self.image)
        input_bbox = np.array(bbox_coords[k])
        self.masks_tuned, _, _ = self.predictor_tuned.predict(
            point_coords=None,
            box=input_bbox,
            multimask_output=False,
        )
        self.masks_orig, _, _ = self.predictor_original.predict(
        point_coords=None,
        box=input_bbox,
        multimask_output=False,
        )
    @staticmethod
    def _show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    @staticmethod
    def _show_box(box, ax):
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))

    def compare_masks(self):
        _, axs = plt.subplots(1, 2, figsize=(25, 25))
        axs[0].imshow(self.image)
        self._show_mask(self.masks_tuned, axs[0])
        self._show_box(input_bbox, axs[0])
        axs[0].set_title('Mask with Tuned Model', fontsize=26)
        axs[0].axis('off')
        axs[1].imshow(self.image)
        self._show_mask(self.masks_orig, axs[1])
        self._show_box(input_bbox, axs[1])
        axs[1].set_title('Mask with Untuned Model', fontsize=26)
        axs[1].axis('off')
        plt.show()
