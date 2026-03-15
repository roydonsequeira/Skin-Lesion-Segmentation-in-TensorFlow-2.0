import os
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

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  # (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x  # (1, 256, 256, 3)

def save_results(ori_x, y_pred, save_image_path):
    # Save the result image
    cv2.imwrite(save_image_path, y_pred * 255)

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

def main():
    # Custom CSS to add background image
    bg_image_path = "Gemini_Generated_Image.jpeg"  # Use a relative path
    
    # Display background image using st.markdown
    st.markdown(
        f"""
        <style>
            body {{
                background-image: url("{bg_image_path}");
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Medical image segmentation: Skin Lesion")

    np.random.seed(42)
    torch.manual_seed(42)

    create_dir("customs")

    model = build_unet((3, H, W))  # Adjust this based on your PyTorch model
    model.load_state_dict(torch.load("files/model.pth"))  # Assuming you saved the PyTorch model using torch.save

    st.sidebar.title("Custom Input")
    custom_image = st.sidebar.file_uploader("Drag and drop your custom image here", type=["jpg", "png"])

    if custom_image:
        """ Read the image """
        ori_x = Image.open(custom_image)
        x = np.array(ori_x)
        x = cv2.resize(x, (W, H)) / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Predicting the mask """
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2)  # (1, 3, 256, 256)
            y_pred = model(x_tensor).cpu().numpy()[0] > 0.5
            y_pred = np.squeeze(y_pred, axis=0)
            y_pred = y_pred.astype(np.int32)

        """ Saving the predicted mask """
        save_image_path = os.path.join("customs", "custom_input_result.png")
        save_results(np.array(ori_x), y_pred, save_image_path)

        """ Display Result """
        st.image(ori_x, caption="Original Image", use_column_width=True)
        st.image(y_pred * 255, caption="Predicted Mask", use_column_width=True)
        st.success("Prediction complete. Check the 'customs' folder for the saved image.")

        """ Calculate and display lesion dimensions """
        dimensions = calculate_lesion_dimensions(y_pred)
        display_dimensions(dimensions, units="pixels")

if __name__ == "__main__":
    main()