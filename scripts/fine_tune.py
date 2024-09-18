import os
import cv2
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from datetime import date
from statistics import mean
from omegaconf import DictConfig
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import threshold, normalize
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

log = logging.getLogger(__name__)

device = "cuda"

class FineTuner:
    def __init__(self, cfg:DictConfig, model_type, checkpoint, device='cuda') -> None:
        """
        Initializes an instance of the class.

        Parameters:
            - cfg (DictConfig): The configuration dictionary.
            - model_type (str): The type of the model.
            - checkpoint (str): The path to the checkpoint file.
            - device (str): The device to use for training (default: 'cuda').
        Returns: 
            None
        """
        log.info(f"Initializing the FineTuner class.")

        # Model Configuration
        self.model_type = model_type                            # Defines the architecture of the SAM model (e.g., 'vit_b', 'vit_l').
        self.device = device                                    # Device on which the model will run ('cuda' for GPU or 'cpu').

        # Training Parameters
        self.optimizer = None                                   # Placeholder for the optimizer, initialized during the training setup.
        self.loss_fn = None                                     # Placeholder for the loss function, to be assigned during training.
        self.num_epochs = 20 # put in config                  # Number of training epochs, set to 100 by default.
        self.losses = []                                        # List to track the loss values across each epoch during training.

        # Data Storage and Management
        self.transformed_data = defaultdict(dict)               # Stores transformed data for training/fine-tuning (defaultdict of dictionaries).
        self.keys = None
        self.masks_dir = Path(cfg.data.masks_dir)               # Placeholder for data keys, populated after loading the dataset.
        self.bbox_coords_dir = Path(cfg.data.bbox_dir)          # Bounding box directory loaded from the configuration file (cfg).
        self.pixel_bbox_coords = self.configure_bbox_coords()   # Bounding box coordinates in pixel format.

        # SAM model setup
        self.sam_model = sam_model_registry[self.model_type](checkpoint=cfg.data.sam_checkpoint)
        self.sam_model.to(self.device)
        self.sam_model.train()

        self.report_dir = Path(cfg.report.report_dir) / str(date.today())
        os.makedirs(self.report_dir, exist_ok=True)

    def load_cutout_masks(self, cfg:DictConfig) -> dict:
        """
        Loads cutout masks from the specified directory.

        Args:
            cfg (DictConfig): Configuration object containing the directory paths.
        
        Returns:
            dict: A dictionary where keys are image identifiers and values are boolean masks.
        """
        log.info(f"Loading cutout masks from the specified directory.")

        cutout_masks = {}
        for k in self.pixel_bbox_coords.keys():
            k = k.split('_')[0]  # Extract the base image identifier
            # Load the grayscale mask image
            cutout_grayscale = cv2.imread(f"{self.masks_dir}/{k}_mask.jpg", cv2.IMREAD_GRAYSCALE)
            # Convert grayscale mask to boolean mask (black is mask, white is non-mask)
            cutout_masks[k] = (cutout_grayscale == 0)

        return cutout_masks

    def configure_bbox_coords(self) -> dict:
        """
        Converts normalized bounding box coordinates (YOLO format) into pixel coordinates
        to work with the SAM prompt.

        Returns:
            dict: A dictionary where the keys are image identifiers, and the values are 
                bounding box coordinates in pixel format.
        """
        log.info(f"Converting normalized bounding box coordinates to pixel coordinates.")

        # Dimensions of the images (height and width) that will be used to scale the bounding box coordinates.

        ##### These values should be read from the image metadata in the future. #####
        image_height = 6368  # put in config
        image_width = 9592   # put in config

        # Dictionary to hold the normalized bounding box coordinates.
        normalized_bbox_coords = {}

        # Iterating through the bounding box coordinate files in the directory.
        for bbox_file in Path(self.bbox_coords_dir).iterdir():
            with open(bbox_file, 'r') as f:
                lines = f.readlines()
                # Extracting the image identifier from the file name.
                k = bbox_file.stem.split('_')[0]
                # Parsing the normalized coordinates from the first line of the file.
                coords = [float(x) for x in lines[0].strip().split()[1:]]
                normalized_bbox_coords[k] = np.array(coords)

        # Creating a copy to store the converted pixel bounding box coordinates.
        pixel_bbox_coords = normalized_bbox_coords.copy()

        # Loop through each set of normalized coordinates and convert them to pixel values.
        for key, value in normalized_bbox_coords.items():
            pixel_bbox_coords[key] = self._convert_bbox_yolo_to_pixel(value, image_width, image_height)

        return pixel_bbox_coords

    def preprocess_images(self, cfg:DictConfig) -> None:
        """
        Preprocesses the images for fine-tuning. This method makes sure that the images are preprocessed for SAM's arctitecture.
        Args:
            cfg (DictConfig): Configuration dictionary.
            bbox_coords (dict): Dictionary containing bounding box coordinates.
        Returns:
            None
        """
        log.info(f"Preprocessing images for fine-tuning.")
        
        for k in self.pixel_bbox_coords.keys():
            # Read the image from the specified directory
            image_path = f"{cfg.data.images_dir}/{k}.jpg"
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Apply resizing transformation to the image
            transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
            input_image = transform.apply_image(image)

            # Convert the transformed image to a PyTorch tensor and adjust its shape
            input_image_torch = torch.as_tensor(input_image, device=self.device)
            transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            # Preprocess the transformed image using the SAM model's preprocessing pipeline
            input_image = self.sam_model.preprocess(transformed_image)

            # Store the original and transformed image sizes for later use
            original_image_size = image.shape[:2]  # Original image size (height, width)
            input_size = tuple(transformed_image.shape[-2:])  # Transformed image size (height, width)

            # Save the processed image and size information in a dictionary
            self.transformed_data[k]['image'] = input_image
            self.transformed_data[k]['input_size'] = input_size
            self.transformed_data[k]['original_image_size'] = original_image_size
  
    def setup_optimizer(self, lr=1e-4, wd=0) -> None:  # put (lr=1e-4, wd=0) in config
        """
        Sets up the optimizer for the SAM model's mask decoder.
        Parameters:
        - lr (float): The learning rate for the optimizer. Default is 1e-4.
        - wd (float): The weight decay for the optimizer. Default is 0.
        Returns:
        - None
        """
        self.optimizer = torch.optim.Adam(self.sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
  
    def setup_loss_function(self) -> None:
        """
        Set up the loss function for the model.
        Parameters:
            loss_fn (callable): The loss function to be used.
        Returns:
            None
        """
        # self.loss_fn = torch.nn.MSELoss() # Mean Squared Error Loss
        self.loss_fn = torch.nn.BCELoss() # Binary Cross Entropy Loss
        self.keys = list(self.pixel_bbox_coords.keys())
  
    def run_fine_tuning(self):
        """
        Fine-tunes the SAM model over multiple epochs.
    
        For each epoch, the method processes each image by generating embeddings, 
        decoding masks, and computing the loss between the predicted and ground truth masks. 
        Gradients are updated, and the loss is recorded.

        Parameters:
        - self: The instance of the class.

        Returns:
        - None
        """
        log.info(f"Fine-tuning the model over {self.num_epochs} epochs.")
        
        cutout_masks = self.load_cutout_masks(cfg=DictConfig)

        # Progress bar for epochs
        for epoch in tqdm(range(self.num_epochs), desc="Epoch Progress", unit="epoch"):
            epoch_losses = []  # Store losses for the current epoch

            # Progress bar for image processing
            for k in tqdm(self.keys, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False, unit="image"):
                # Get preprocessed image, input size, and original image size
                input_image = self.transformed_data[k]['image'].to(self.device)
                input_size = self.transformed_data[k]['input_size']
                original_image_size = self.transformed_data[k]['original_image_size']

                # To keep the image encoder and prompt fixed during the training
                with torch.no_grad(): 
                    # Generate image embeddings using the SAM model's image encoder
                    image_embedding = self.sam_model.image_encoder(input_image)

                    transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

                    # Fetch and transform the bounding box coordinates
                    prompt_box = np.array(self.pixel_bbox_coords[k])
                    box = transform.apply_boxes(prompt_box, original_image_size)
                    box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)[None, :] #convert bboxes to tensor

                    # Generate sparse and dense embeddings using the prompt encoder
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None,
                    )

                # Decode the mask using the SAM model's mask decoder
                low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # Post-process the masks and upscale them to match the original image size
                upscaled_masks = self.sam_model.postprocess_masks(
                    low_res_masks, input_size, original_image_size
                ).to(self.device)

                # Convert the masks to binary form
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

                # Resize the ground truth mask to match the dimensions and convert to tensor
                cutout_mask_resized = torch.from_numpy(
                    np.resize(cutout_masks[k], (1, 1, cutout_masks[k].shape[0], cutout_masks[k].shape[1]))
                ).to(self.device)
                cutout_binary_mask = torch.as_tensor(cutout_mask_resized > 0, dtype=torch.float32)

                # Calculate the loss between the predicted binary mask and the ground truth binary mask
                loss = self.loss_fn(binary_mask, cutout_binary_mask)

                # Zero the gradients, perform backpropagation, and update model parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Store the loss for the current image
                epoch_losses.append(loss.item())

            # Append the losses for the current epoch and log the mean loss
            self.losses.append(epoch_losses)
            print(f"EPOCH: {epoch + 1}")
            print(f"Mean loss: {mean(epoch_losses)}")
        
        self.save_model(os.path.join(self.report_dir, "fine_tuned_sam_model.pth"))
  
    def plot_mean_epoch_loss(self):
        """
        Plots the mean epoch loss.
        This method calculates the mean loss for each epoch and plots it on a graph.
        The x-axis represents the epoch number, while the y-axis represents the loss value.
        Returns:
            None
        """
        log.info(f"Plotting and saving the mean epoch loss.")

        # Create a folder with the current date inside cfg.report directory
        mean_losses = [mean(x) for x in self.losses]
        plt.plot(list(range(len(mean_losses))), mean_losses)
        plt.title('Mean epoch loss')
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.report_dir, 'mean_epoch_loss.png'))

    def save_model(self, save_path: str) -> None:
        """
        Saves the fine-tuned model to the specified path.
        
        Parameters:
            - save_path (str): The path where the model will be saved.
        """
        log.info(f"Saving the fine-tuned model to {save_path}.")
        torch.save(self.sam_model.state_dict(), save_path)

    @staticmethod
    def _convert_bbox_yolo_to_pixel(coords, img_width, img_height):
        """
        Convert bounding box coordinates from YOLO format to pixel format.
        Args:
            coords (tuple): The bounding box coordinates in YOLO format (x_center, y_center, width, height).
            img_width (int): The width of the image.
            img_height (int): The height of the image.
        Returns:
            tuple: The bounding box coordinates in pixel format (x0, y0, x1, y1).
        """
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

def main(cfg:DictConfig) -> None:
    """
    Main function for fine-tuning a model using the provided configuration.
    Args:
        cfg (DictConfig): Configuration dictionary containing the necessary parameters.
    Returns:
        None
    """
    log.info(f"Fine-tuning the model using the provided configuration.")

    fine_tuner = FineTuner(cfg, model_type='vit_b', checkpoint=cfg.data.sam_checkpoint, device='cuda')
    fine_tuner.preprocess_images(cfg)
    fine_tuner.setup_optimizer(lr=1e-4, wd=0)
    fine_tuner.setup_loss_function()
    fine_tuner.run_fine_tuning()
    fine_tuner.plot_mean_epoch_loss()