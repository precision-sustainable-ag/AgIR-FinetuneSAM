Fine tuning Sam for Ag Image repo:
* Check this link for details: https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/

Summary:
1. Why finetune: The purpose of fine-tuning a model is to obtain higher performance on data that the pre-trained model has not seen before.

2. Focus on finetuning Mask Decoder part of the SAM. Reason:
    a. Mask decoder is lightweight and therefore easier, faster, and more memory efficient to fine-tune

3. We cannot use SamPredictor.predict (link) for two reasons:
    a. We want to fine-tune only the mask decoder
    b. This function calls SamPredictor.predict_torch which has the  @torch.no_grad() decorator, which prevents us from computing gradients

4. Need to create a custom-dataset:
    We need three things to fine-tune our model:
        a. Images 
        b. Segmentation masks
        c. Prompts: bounding boxes

5. Input data-preprocessing:
    This step is necessary to make our custom-dataset readable by SAM. We need to preprocess the scans from numpy arrays to pytorch tensors.

    a. Follow what happens inside SamPredictor.set_image (https://github.com/facebookresearch/segment-anything/blob/c1910835a32a05cbb79bdacbec8f25914a7e3a20/segment_anything/predictor.py#L34-L60) and SamPredictor.set_torch_image (https://github.com/facebookresearch/segment-anything/blob/c1910835a32a05cbb79bdacbec8f25914a7e3a20/segment_anything/predictor.py#L63) which preprocesses the image. 

    b. Use utils.transform.ResizeLongestSide to resize the image, as this is the transformer used inside the predictor (https://github.com/facebookresearch/segment-anything/blob/c1910835a32a05cbb79bdacbec8f25914a7e3a20/segment_anything/predictor.py#L31). We can then convert the image to a pytorch tensor and use the SAM preprocess method (https://github.com/facebookresearch/segment-anything/blob/c1910835a32a05cbb79bdacbec8f25914a7e3a20/segment_anything/modeling/sam.py#L164) to finish preprocessing.

6. Training setup:
    a. Download the model checkpoint for the vit_b model and load them in:
        sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')

    b. Set up an Adam optimizer with defaults and specify that the parameters to tune are those of the mask decoder:
        optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters()) 

    c. Set up our loss function:
        loss_fn = torch.nn.BCELoss() # Binary Cross Entropy Loss

7. Training loop:
    a. Use GPU "cuda"

    b. Wrap sam_model.image_encoder and sam_model.prompt_encoder in torch.no_grad() context manager.        
        Reason:
            - we are not looking to fine-tune image_encoder or prompt_encoder
    
    c. Generate the masks

    d. Upscale the masks back to the original image size since they are low resolution with Sam.postprocess_masks 

    e. Calculate the loss and run an optimization step

    f. Save the fine-tuned model
