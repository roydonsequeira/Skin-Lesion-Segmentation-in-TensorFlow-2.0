import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from model import build_unet  # Adjust this import based on your PyTorch model
from train import load_data, create_dir

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

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (H, W)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x / 255.0
    x = x.astype(np.int32)  # (256, 256)
    return ori_x, x

def save_results(ori_x, ori_y, y_pred, save_image_path):
    # Create a line to separate images horizontally
    line = np.ones((H, 10, 3)) * 255

    # Expand dimensions for proper concatenation
    ori_y = np.expand_dims(ori_y, axis=-1)  # (256, 256, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)  # (256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)  # (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  # (256, 256, 3)

    # Concatenate along width to display the images horizontally
    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred * 255], axis=1)

    # Save the result image
    cv2.imwrite(save_image_path, cat_images)

def calculate_lesion_dimensions(y_pred):
    """Calculate lesion dimensions"""
    # Convert to uint8 and find contours in the predicted mask
    y_pred_uint8 = (y_pred * 255).astype(np.uint8)  # Convert to uint8
    _, binary_mask = cv2.threshold(y_pred_uint8, 128, 255, cv2.THRESH_BINARY)  # Ensure binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # No lesion detected

    # Assuming there's only one lesion, choose the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box of the lesion
    x, y, width, height = cv2.boundingRect(largest_contour)

    # Calculate the area of the lesion using the contour area
    area = cv2.contourArea(largest_contour)

    # Return the dimensions as a dictionary
    dimensions = {'x': x, 'y': y, 'width': width, 'height': height, 'area': area}
    return dimensions


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    torch.manual_seed(42)

    """ Folder for saving results """
    create_dir("results")

    """ Load the model """
    model = build_unet((3, H, W))  # Adjust this based on your PyTorch model
    model.load_state_dict(torch.load("files/model.pth"))  # Assuming you saved the PyTorch model using torch.save

    """ Load the test data """
    dataset_path = "C:\\Users\\roy\\OneDrive\\Desktop\\FYP"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracting the image name """
        name = os.path.split(x)[-1]

        """ Read the image and mask """
        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)

        """ Predicting the mask """
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2)  # (1, 3, 256, 256)
            y_pred = model(x_tensor).cpu().numpy()[0] > 0.5
            y_pred = np.squeeze(y_pred, axis=0)
            y_pred = y_pred.astype(np.int32)

        """ Saving the predicted mask """
        save_image_path = os.path.join("results", os.path.splitext(name)[0] + ".png")
        save_results(ori_x, ori_y, y_pred, save_image_path)

        """ Flatten the array """
        y = y.flatten()
        y_pred = y_pred.flatten()

        """ Calculating metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

        """ Calculate and print lesion dimensions """
        dimensions = calculate_lesion_dimensions(y_pred)
        if dimensions:
            print(f"Lesion Dimensions for {name}: {dimensions}")

    """ mean metrics values """
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image Name", "Acc", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/score.csv")
