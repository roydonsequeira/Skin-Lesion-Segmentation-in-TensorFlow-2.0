import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image
from model import build_unet  # Adjust this import based on your PyTorch model
from train import create_dir
from eval import calculate_lesion_dimensions

H = 256
W = 256

def read_image(image_data):
    image = Image.open(image_data)
    image = np.array(image)
    image = cv2.resize(image, (W, H))
    return image

def overlay_masks(image, mask1, mask2):
    # Resize masks to match the shape of the original image
    resized_mask1 = cv2.resize(mask1, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    resized_mask2 = cv2.resize(mask2, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert masks to 3-channel format for overlaying with the original image
    resized_mask1 = cv2.merge([resized_mask1, resized_mask1, resized_mask1])
    resized_mask2 = cv2.merge([resized_mask2, resized_mask2, resized_mask2])
    
    # Overlay masks on the original image
    overlay1 = cv2.addWeighted(image, 0.5, resized_mask1, 0.5, 0)
    overlay2 = cv2.addWeighted(image, 0.5, resized_mask2, 0.5, 0)
    result = cv2.addWeighted(overlay1, 0.5, overlay2, 0.5, 0)
    
    return result




def main():
  
    st.title("Medical image segmentation: Skin Lesion")

    np.random.seed(42)
    torch.manual_seed(42)

    create_dir("customs")

    model = build_unet((3, H, W))  # Adjust this based on your PyTorch model
    model.load_state_dict(torch.load("files/model.pth"))  # Assuming you saved the PyTorch model using torch.save

    st.sidebar.title("Custom Input")
    baseline_image = st.sidebar.file_uploader("Upload Baseline Image", type=["jpg", "png"])
    current_image = st.sidebar.file_uploader("Upload Current Image", type=["jpg", "png"])

    if baseline_image and current_image:
        """ Read the images """
        base_image = read_image(baseline_image)
        current_image = read_image(current_image)

        """ Predicting the mask for baseline image """
        with torch.no_grad():
            base_image_tensor = torch.from_numpy(base_image / 255.0).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, 256, 256)
            base_mask = model(base_image_tensor).cpu().numpy()[0] > 0.5
            base_mask = np.squeeze(base_mask, axis=0)
            base_mask = base_mask.astype(np.uint8) * 255

        """ Predicting the mask for current image """
        with torch.no_grad():
            current_image_tensor = torch.from_numpy(current_image / 255.0).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, 256, 256)
            current_mask = model(current_image_tensor).cpu().numpy()[0] > 0.5
            current_mask = np.squeeze(current_mask, axis=0)
            current_mask = current_mask.astype(np.uint8) * 255

        """ Display Results """
        st.image(overlay_masks(base_image, base_mask, current_mask), caption="Comparison of Baseline and Current Masks", use_column_width=True)
        st.success("Prediction complete.")

if __name__ == "__main__":
    main()