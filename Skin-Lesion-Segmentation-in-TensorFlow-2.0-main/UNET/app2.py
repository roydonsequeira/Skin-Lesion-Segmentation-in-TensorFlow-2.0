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
    image = cv2.resize(image, (W, H)) / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def display_dimensions(dimensions, units):
    if dimensions:
        st.sidebar.subheader("Lesion Dimensions:")
        st.sidebar.text(f"X: {dimensions['x']} {units}")
        st.sidebar.text(f"Y: {dimensions['y']} {units}")
        st.sidebar.text(f"Width: {dimensions['width']} {units}")
        st.sidebar.text(f"Height: {dimensions['height']} {units}")
        st.sidebar.text(f"Area: {dimensions['area']} {units}²")
    else:
        st.sidebar.warning("No lesion detected.")

def compare_dimensions(base_dimensions, current_dimensions):
    if base_dimensions and current_dimensions:
        area_difference = current_dimensions['area'] - base_dimensions['area']
        if area_difference > 0:
            st.sidebar.text("Lesion dimension increased.")
        elif area_difference < 0:
            st.sidebar.text("Lesion dimension decreased.")
        else:
            st.sidebar.text("Lesion dimension remained unchanged.")

def overlay_masks(image, mask1, mask2):
    # Resize masks to match the shape of the original image
    resized_mask1 = cv2.resize(mask1, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    resized_mask2 = cv2.resize(mask2, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    
    # Convert masks to 3-channel format for overlaying with the original image
    resized_mask1 = cv2.merge([resized_mask1, resized_mask1, resized_mask1])
    resized_mask2 = cv2.merge([resized_mask2, resized_mask2, resized_mask2])
    
    # Overlay masks on the original image
    overlay1 = cv2.addWeighted(image[0], 0.5, resized_mask1.astype(np.float32), 0.5, 0, dtype=cv2.CV_32F)
    overlay2 = cv2.addWeighted(image[0], 0.5, resized_mask2.astype(np.float32), 0.5, 0, dtype=cv2.CV_32F)
    result = cv2.addWeighted(overlay1, 0.5, overlay2, 0.5, 0, dtype=cv2.CV_32F)
    
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
            base_image_tensor = torch.from_numpy(base_image).permute(0, 3, 1, 2)  # (1, 3, 256, 256)
            base_mask = model(base_image_tensor).cpu().numpy()[0] > 0.5
            base_mask = np.squeeze(base_mask, axis=0)
            base_mask = base_mask.astype(np.int32)

        """ Predicting the mask for current image """
        with torch.no_grad():
            current_image_tensor = torch.from_numpy(current_image).permute(0, 3, 1, 2)  # (1, 3, 256, 256)
            current_mask = model(current_image_tensor).cpu().numpy()[0] > 0.5
            current_mask = np.squeeze(current_mask, axis=0)
            current_mask = current_mask.astype(np.int32)

        """ Display Results """
        st.image(base_image, caption="Baseline Image", use_column_width=True)
        st.image(base_mask * 255, caption="Baseline Predicted Mask", use_column_width=True)
        st.image(current_image, caption="Current Image", use_column_width=True)
        st.image(current_mask * 255, caption="Current Predicted Mask", use_column_width=True)
        st.success("Prediction complete.")
        st.image(overlay_masks(base_image, base_mask, current_mask), caption="Comparison of Baseline and Current Masks", use_column_width=True)

        """ Calculate and display lesion dimensions """
        base_dimensions = calculate_lesion_dimensions(base_mask)
        current_dimensions = calculate_lesion_dimensions(current_mask)
        display_dimensions(base_dimensions, units="pixels")
        display_dimensions(current_dimensions, units="pixels")

        """ Compare lesion dimensions """
        compare_dimensions(base_dimensions, current_dimensions)

if __name__ == "__main__":
    main()
