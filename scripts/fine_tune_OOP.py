import os
import cv2
import uuid
import torch
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datetime import date
from statistics import mean
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import threshold, normalize
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


# Configure logging
log = logging.getLogger(__name__)

# Set device to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"


class SAMDataset(Dataset):
    def __init__( self, cfg, pixel_bbox_coords: dict, masks_dir: str, images_dir: str, sam_model):
        """
        Custom dataset to handle the loading of images, bounding boxes, and masks.

        Args:
            cfg: Configuration object containing dataset parameters.
            pixel_bbox_coords (dict): Dictionary mapping image keys to bounding box coordinates.
            masks_dir (str): Directory path where mask images are stored.
            images_dir (str): Directory path where input images are stored.
            sam_model: SAM model instance used for image preprocessing.
        """
        self.cfg = cfg
        self.pixel_bbox_coords = pixel_bbox_coords
        self.masks_dir = masks_dir
        self.images_dir = images_dir
        self.sam_model = sam_model
        self.keys = list(self.pixel_bbox_coords.keys())

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.keys)

    def __getitem__(self, idx: int):
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Preprocessed input image tensor.
                - Bounding box tensor.
                - Cutout mask tensor.
                - Original image size tensor.
                - Input image size tensor.
        """
        k = self.keys[idx]

        # Load image
        image_path = f"{self.images_dir}/{k}.jpg"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply resizing transformation to the image
        transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device).permute(2, 0, 1).contiguous()

        # Preprocess image (now it's 4D, as a batch of 1)
        input_image = self.sam_model.preprocess(input_image_torch[None, :, :, :])

        original_image_size = image.shape[:2]  # (height, width)
        input_size = input_image_torch.shape[-2:]

        # Load mask
        mask_path = f"{self.masks_dir}/{k}.png"
        gt_grayscale = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Create a cutout mask where 255 becomes 0 and others become 1
        cutout_mask = np.where(gt_grayscale == 255, 0, 1)

        # Convert cutout mask to tensor and add necessary dimensions
        cutout_mask_resized = torch.from_numpy(
            cutout_mask
        ).unsqueeze(0).unsqueeze(0).float()  # Shape: [1, 1, H, W]

        # Fetch and transform the bounding box coordinates
        prompt_box = np.array(self.pixel_bbox_coords[k])
        box = transform.apply_boxes(prompt_box, image.shape[:2])
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)[None, :]  # Shape: [1, 4]

        return input_image, box_torch, cutout_mask_resized, torch.tensor(original_image_size), torch.tensor(input_size)


def save_masks(pred_mask: torch.Tensor, cutout_binary_mask: torch.Tensor, batch_idx: int, save_dir: str = ".") -> None:
    """
    Save the predicted mask and ground truth mask with a unique identifier.

    Args:
        pred_mask (torch.Tensor): The predicted mask from the model, shape [1, 1, H, W].
        cutout_binary_mask (torch.Tensor): The ground truth binary mask, shape [1, 1, H, W].
        batch_idx (int): The current batch index, used for tracking the batch.
        save_dir (str, optional): Directory where masks will be saved. Default is current directory.

    Returns:
        None
    """
    # Generate a short UUID for the file names
    unique_id = str(uuid.uuid4())[:8]  # First 8 characters

    # Convert predicted mask to numpy array
    pred_mask_np = pred_mask.cpu().numpy()[0, 0]  # Shape: [H, W]

    # Convert ground truth (cutout) mask to numpy array
    cutout_mask_np = cutout_binary_mask.cpu().numpy()[0, 0]  # Shape: [H, W]

    # Optional: Save comparison image using matplotlib
    plt.figure(figsize=(10, 5))

    # Display predicted mask
    plt.subplot(1, 2, 1)
    plt.imshow(pred_mask_np, cmap='gray')
    plt.title("Predicted Mask")

    # Display ground truth mask
    plt.subplot(1, 2, 2)
    plt.imshow(cutout_mask_np, cmap='gray')
    plt.title(f"Ground Truth Mask: {np.unique(cutout_mask_np)}")

    plt.tight_layout()
    comparison_path = f"{save_dir}/{unique_id}_batch{batch_idx}_comparison.png"
    plt.savefig(comparison_path)
    plt.close()

    log.info(f"Predicted mask comparison saved at: {comparison_path}")


class FineTuner:
    def __init__(self, cfg, model_type: str, checkpoint: str, device: str = 'cuda'):
        """
        Initializes the FineTuner with configuration, model, optimizer, and loss function.

        Args:
            cfg: Configuration object containing training parameters.
            model_type (str): Type of SAM model to use (e.g., 'vit_b').
            checkpoint (str): Path to the model checkpoint for initialization.
            device (str, optional): Device to run the model on. Defaults to 'cuda'.
        """
        log.info("Initializing the FineTuner class.")
        self.cfg = cfg
        self.model_type = model_type
        self.device = device
        self.optimizer = None
        self.loss_fn = None
        self.num_epochs = cfg.experiments.epochs
        self.save_every_n_epochs = cfg.experiments.save_every_n_epochs
        self.val_every_n_epochs = cfg.experiments.val.val_every_n_epochs
        self.losses = []
        self.validation_losses = []

        # Initialize SAM model
        self.sam_model = sam_model_registry[self.model_type](checkpoint=checkpoint)
        self.sam_model.to(self.device)
        self.sam_model.train()

        # Set up report directory with current date
        self.report_dir = Path(cfg.report.report_dir) / str(date.today())
        os.makedirs(self.report_dir, exist_ok=True)

    def configure_bbox_coords(self, cfg, bbox_dir: str) -> dict:
        """
        Converts normalized bounding box coordinates to pixel coordinates.

        Args:
            cfg: Configuration object containing dataset parameters.
            bbox_dir (str): Directory containing bounding box files.

        Returns:
            dict: Dictionary mapping image keys to pixel bounding box coordinates.
        """
        log.info("Converting normalized bounding box coordinates to pixel coordinates.")

        # Image dimensions (should ideally come from cfg)
        image_height = 6368  # TODO: Move to configuration
        image_width = 9592   # TODO: Move to configuration

        normalized_bbox_coords = {}

        # Read bounding box files and store normalized coordinates
        for bbox_file in Path(bbox_dir).iterdir():
            with open(bbox_file, 'r') as f:
                lines = f.readlines()
                k = bbox_file.stem.split('_')[0]
                coords = [float(x) for x in lines[0].strip().split()[1:]]
                normalized_bbox_coords[k] = np.array(coords)

        # Convert normalized coordinates to pixel coordinates
        pixel_bbox_coords = {}
        for key, value in normalized_bbox_coords.items():
            pixel_bbox_coords[key] = self._convert_bbox_yolo_to_pixel(value, image_width, image_height)

        return pixel_bbox_coords

    def setup_dataloader(self, cfg) -> None:
        """
        Sets up the PyTorch DataLoader for training and validation datasets.

        Args:
            cfg: Configuration object containing dataset parameters.

        Returns:
            None
        """
        # Training data loader
        train_pixel_bbox_coords = self.configure_bbox_coords(cfg, cfg.experiments.train.train_bbox_dir)
        train_dataset = SAMDataset(
            cfg,
            train_pixel_bbox_coords,
            cfg.experiments.train.train_mask_dir,
            cfg.experiments.train.train_image_dir,
            self.sam_model
        )
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.experiments.train.batch_size,
            shuffle=True,
            num_workers=cfg.experiments.train.num_workers
        )

        # Validation data loader
        val_pixel_bbox_coords = self.configure_bbox_coords(cfg, cfg.experiments.val.val_bbox_dir)
        val_dataset = SAMDataset(
            cfg,
            val_pixel_bbox_coords,
            cfg.experiments.val.val_mask_dir,
            cfg.experiments.val.val_image_dir,
            self.sam_model
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.experiments.val.batch_size,
            shuffle=False,
            num_workers=cfg.experiments.val.num_workers  # Added num_workers for consistency
        )

    def setup_optimizer(self) -> None:
        """
        Configures the optimizer for training.

        Returns:
            None
        """
        self.optimizer = torch.optim.Adam(
            self.sam_model.mask_decoder.parameters(),
            lr=self.cfg.experiments.train.lr,
            weight_decay=self.cfg.experiments.train.weight_decay
        )
        log.info("Optimizer has been set up.")

    def setup_loss_function(self) -> None:
        """
        Configures the loss function for training.

        Returns:
            None
        """
        self.loss_fn = torch.nn.BCELoss()
        log.info("Loss function has been set up.")

    def compute_iou(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
        """
        Compute the Intersection over Union (IoU) between the predicted mask and the target mask.

        Args:
            pred_mask (torch.Tensor): Predicted binary mask.
            target_mask (torch.Tensor): Ground truth binary mask.

        Returns:
            float: Mean IoU score across the batch.
        """
        pred_mask = pred_mask > 0.5  # Convert predictions to binary
        target_mask = target_mask > 0.5  # Ensure ground truth is binary

        intersection = (pred_mask & target_mask).float().sum((1, 2))  # Intersection
        union = (pred_mask | target_mask).float().sum((1, 2))  # Union

        iou = intersection / (union + 1e-6)  # Avoid division by zero
        return iou.mean().item()  # Mean IoU across batch

    def run_fine_tuning(self) -> None:
        """
        Executes the fine-tuning process over the specified number of epochs.
        This includes training, validation, loss computation, IoU calculation, and model checkpointing.

        Returns:
            None
        """
        log.info(f"Starting fine-tuning for {self.num_epochs} epochs.")

        for epoch in range(self.num_epochs):
            epoch_losses = []
            epoch_ious = []  # Track IoUs for each batch

            # Set model to training mode
            self.sam_model.train()

            batch_idx = 0
            save_comparison = False  # Flag to control mask saving

            # Iterate over training DataLoader
            for batch in tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs} (Training)"
            ):
                input_images, boxes, cutout_masks, original_image_sizes, input_sizes = batch
                log.debug(f"Batch {batch_idx}:")
                log.debug(f" - Input images shape: {input_images.shape}")
                log.debug(f" - Boxes shape: {boxes.shape}")
                log.debug(f" - Cutout masks shape: {cutout_masks.shape}")
                log.debug(f" - Original image sizes: {original_image_sizes}")
                log.debug(f" - Input sizes: {input_sizes}")

                # Move input images to the specified device
                input_images = input_images.to(self.device)

                batch_loss = 0.0  # Initialize batch loss
                batch_iou = 0.0   # Initialize batch IoU

                # Process each image in the batch individually
                for idx in range(input_images.shape[0]):
                    with torch.no_grad():
                        # Generate image embeddings using the SAM model's image encoder
                        image_embedding = self.sam_model.image_encoder(input_images[idx])
                        log.debug(f"Image {idx + 1}: Generated image embeddings.")

                        # Generate sparse and dense embeddings using the prompt encoder
                        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                            points=None,
                            boxes=boxes[idx].unsqueeze(0).to(self.device),
                            masks=None,
                        )
                        log.debug(f"Image {idx + 1}: Generated prompt embeddings.")

                    # Decode the mask for the current image
                    low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    log.debug(f"Image {idx + 1}: Decoded low-resolution masks.")

                    # Retrieve original image size
                    original_size = tuple(original_image_sizes[idx].tolist())
                    log.debug(f"Image {idx + 1}: Original size {original_size}.")

                    # Postprocess masks to match original image dimensions
                    upscaled_masks = self.sam_model.postprocess_masks(
                        low_res_masks, input_sizes[idx], original_size
                    )
                    log.debug(f"Image {idx + 1}: Upscaled masks shape {upscaled_masks.shape}.")

                    # Normalize and threshold masks
                    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
                    cutout_binary_mask = cutout_masks[idx].to(self.device)

                    # Compute loss between predicted mask and ground truth
                    image_loss = self.loss_fn(binary_mask, cutout_binary_mask)
                    batch_loss += image_loss

                    # Compute IoU and accumulate
                    batch_iou += self.compute_iou(binary_mask, cutout_binary_mask)

                    # Optionally save mask comparisons
                    pred_mask = (upscaled_masks > 0.5).float()
                    if save_comparison:
                        save_masks(pred_mask, cutout_binary_mask, batch_idx, save_dir=".")
                        save_comparison = False  # Reset flag

                # Average the accumulated loss and IoU over the batch
                batch_loss /= input_images.shape[0]
                batch_iou /= input_images.shape[0]

                # Backpropagation and optimizer step
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # Log and store the batch loss and IoU
                epoch_losses.append(batch_loss.item())
                epoch_ious.append(batch_iou)
                log.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss={batch_loss.item():.4f}, IoU={batch_iou:.4f}")
                batch_idx += 1

            # Calculate and log average loss and IoU for the epoch
            mean_epoch_loss = mean(epoch_losses)
            mean_epoch_iou = mean(epoch_ious)
            self.losses.append(mean_epoch_loss)
            log.info(f"Epoch {epoch + 1}: Mean Loss={mean_epoch_loss:.4f}, Mean IoU={mean_epoch_iou:.4f}")

            # Perform validation at specified intervals
            if (epoch + 1) % self.val_every_n_epochs == 0:
                self.sam_model.eval()
                val_losses = []
                val_ious = []

                with torch.no_grad():
                    for batch in tqdm(
                        self.val_dataloader,
                        desc=f"Epoch {epoch + 1}/{self.num_epochs} (Validation)"
                    ):
                        input_images, boxes, cutout_masks, original_image_sizes, input_sizes = batch

                        input_images = input_images.to(self.device)
                        batch_loss = 0.0
                        batch_iou = 0.0

                        # Process each image in the validation batch
                        for idx in range(input_images.shape[0]):
                            # Generate image embeddings
                            image_embedding = self.sam_model.image_encoder(input_images[idx])

                            # Generate prompt embeddings
                            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                                points=None,
                                boxes=boxes[idx].unsqueeze(0).to(self.device),
                                masks=None
                            )

                            # Decode masks
                            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                                image_embeddings=image_embedding,
                                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False,
                            )

                            # Postprocess masks
                            upscaled_masks = self.sam_model.postprocess_masks(
                                low_res_masks, input_sizes[idx], tuple(original_image_sizes[idx].tolist())
                            )

                            # Normalize and threshold masks
                            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
                            cutout_binary_mask = cutout_masks[idx].to(self.device)

                            # Compute loss and IoU
                            image_loss = self.loss_fn(binary_mask, cutout_binary_mask)
                            batch_loss += image_loss
                            batch_iou += self.compute_iou(binary_mask, cutout_binary_mask)

                        # Average loss and IoU over the validation batch
                        batch_loss /= input_images.shape[0]
                        batch_iou /= input_images.shape[0]

                        val_losses.append(batch_loss.item())
                        val_ious.append(batch_iou)
                        log.info(f"Validation Batch {batch_idx + 1}: Loss={batch_loss.item():.4f}, IoU={batch_iou:.4f}")

                # Calculate and log average validation loss and IoU
                mean_val_loss = mean(val_losses)
                mean_val_iou = mean(val_ious)
                self.validation_losses.append(mean_val_loss)
                log.info(f"Epoch {epoch + 1} (Validation): Mean Loss={mean_val_loss:.4f}, Mean IoU={mean_val_iou:.4f}")

            # Save the model checkpoint at specified intervals
            if (epoch + 1) % self.save_every_n_epochs == 0:
                save_path = self.report_dir / f"fine_tuned_sam_model_epoch_{epoch + 1}.pth"
                self.save_model(str(save_path))
                log.info(f"Model checkpoint saved at epoch {epoch + 1} to {save_path}.")

        # Save the final fine-tuned model after all epochs
        final_save_path = self.report_dir / "fine_tuned_sam_model.pth"
        self.save_model(str(final_save_path))
        log.info(f"Final fine-tuned model saved to {final_save_path}.")

    def save_model(self, save_path: str) -> None:
        """
        Saves the fine-tuned model's state dictionary to the specified path.

        Args:
            save_path (str): File path where the model state will be saved.

        Returns:
            None
        """
        log.info(f"Saving the fine-tuned model to {save_path}.")
        torch.save(self.sam_model.state_dict(), save_path)

    @staticmethod
    def _convert_bbox_yolo_to_pixel(coords: np.ndarray, img_width: int, img_height: int) -> tuple:
        """
        Converts YOLO-format bounding box coordinates to pixel coordinates.

        Args:
            coords (np.ndarray): YOLO-format bounding box coordinates [x_center, y_center, width, height].
            img_width (int): Width of the image in pixels.
            img_height (int): Height of the image in pixels.

        Returns:
            tuple: Bounding box coordinates in pixel format (x0, y0, x1, y1).
        """
        x_center, y_center, width, height = coords
        x_center_pixel = x_center * img_width
        y_center_pixel = y_center * img_height
        width_pixel = width * img_width
        height_pixel = height * img_height
        x0 = x_center_pixel - width_pixel / 2
        y0 = y_center_pixel - height_pixel / 2
        x1 = x_center_pixel + width_pixel / 2
        y1 = y_center_pixel + height_pixel / 2
        return x0, y0, x1, y1


def main(cfg) -> None:
    """
    Entry point for fine-tuning the SAM model using the provided configuration.

    Args:
        cfg: Configuration object containing all necessary parameters for training.

    Returns:
        None
    """
    log.info("Starting the fine-tuning process with the provided configuration.")
    mp.set_start_method("spawn", force=True)  # Set multiprocessing start method

    # Initialize FineTuner with specified model type and checkpoint
    fine_tuner = FineTuner(
        cfg=cfg,
        model_type=cfg.experiments.model_type,
        checkpoint=cfg.experiments.sam_checkpoint,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Set up DataLoaders, optimizer, and loss function
    fine_tuner.setup_dataloader(cfg)
    fine_tuner.setup_optimizer()
    fine_tuner.setup_loss_function()

    # Begin the fine-tuning process
    fine_tuner.run_fine_tuning()
