import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import numpy as np
import os
import shutil

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ========================== DATASET PREPARATION ============================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.RandomRotation(degrees=(-10, 10)),  # Random rotation between -10 and 10 degrees
])

def visualize_transform(transformed_image):
    plt.imshow(torch.from_numpy(transformed_image))  # Adjust for matplotlib if you included ToTensor()
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
    
def get_hand_points(img):
    results = hands.process(img)
    points = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                points.append([x, y, z])
    else: 
        points = None
    if points is not None and len(points) == 21:
        points = np.array(points)
    return points

def normalize_points(raw_points):
    points = np.copy(raw_points)
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    for i in range(len(points)):
        points[i][0] = (points[i][0] - min_x) / (max_x - min_x)
        points[i][1] = (points[i][1] - min_y) / (max_y - min_y)
    return points
            
def plot_points(points,src_img):
    height = src_img.shape[0]
    width = src_img.shape[1]
    # img = np.copy(src_img)
    img = np.zeros((height, width, 3), np.uint8)
    img.fill(255)
    for hc in mp_hands.HAND_CONNECTIONS:
        cv2.line(img, (int((points[hc[0]][0]) * width), 
                    int((points[hc[0]][1]) * height)),
                    (int((points[hc[1]][0]) * width), 
                    int((points[hc[1]][1]) * height)), (0, 0, 255), 4)
    return img



def transform_image(img,transform): 
    new = transform(img)
    new = (new * 255).byte()
    new = new.permute(1, 2, 0)
    new = new.numpy()
    return new

def create_datapoint(source_path, label, name , dest_path, transform):
    counter = 0
    os.makedirs(dest_path, exist_ok=True)
    os.makedirs(os.path.join(dest_path, label.upper()), exist_ok=True)
    img = cv2.imread(os.path.join(source_path, name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert Original
    raw_points = get_hand_points(img)
    if raw_points is not None:
        normalized_points = normalize_points(raw_points)
        np.save(os.path.join(dest_path, label, name.split('.')[0]), normalized_points)
        counter += 1
    
    # Convert transformed
    transformed_img = transform_image(img, transform)
    transformed_raw_points = get_hand_points(transformed_img)
    if transformed_raw_points is not None: 
        transformed_normalized_points = normalize_points(transformed_raw_points)
        np.save(os.path.join(dest_path, label, name.split('.')[0] + '_transformed'), transformed_normalized_points)
        counter += 1
    return counter
    

def create_dataset(src_path, destination_path, transform):
    count = 0
    letters_converted = []
    for src_path, dirs, files in os.walk(src_path):
        label = src_path.split(os.sep)[-1]
        if label not in letters_converted:
            letters_converted.append(label)
            print(f'Processing {label}')
        for name in files:
            if name == ".DS_Store":
                continue
            c = create_datapoint(src_path, label, name, destination_path, transform)
            count += c

    print(f'Created {count}')
    



# --- REFERENCe FOR HOW TO USE ---
# img = cv2.imread('./inputtest.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# points = get_hand_points(img)
# plotted_img = plot_points(points,img)

# transformed = transform_image(img,transform)

# new_points = get_hand_points(transformed)
# transformed_plot = plot_points(new_points,transformed)
# visualize_transform(img)
# visualize_transform(plotted_img)
# visualize_transform(transformed)
# visualize_transform(transformed_plot)

# create_datapoint('./set2/Train_Alphabet', 'A', '0aff0fc7-568a-40a3-b510-0584d817cd01.rgb_0000.png', './output', transform)

# CREATING DATASET DO NOT UNCOMMENT. 
# create_dataset('./set2/Train_Alphabet',"./landmark_dataset", transform)
# create_dataset('./asl_dataset',"./landmark_dataset2", transform)
# create_dataset("./extras","./landmark_dataset3", transform)


# Function to merge multiple datasets generated
def merge_folders(source, destination):
    for root, dirs, files in os.walk(source):
        # Construct the path relative to the source directory
        rel_path = os.path.relpath(root, source)
        # Determine the destination directory path
        dest_dir = os.path.join(destination, rel_path)
        # Ensure the destination directory exists
        os.makedirs(dest_dir, exist_ok=True)
        for file in files:
            # Source file path
            src_file = os.path.join(root, file)
            # Destination file path
            dest_file = os.path.join(dest_dir, file)
            # Copy the file to the destination directory
            if os.path.exists(dest_file):
                # Optionally handle the case where the file already exists in the destination folder
                # For simplicity, we'll just overwrite it. You might want to check timestamps or sizes before doing so.
                print(f"File {dest_file} will be overwritten by {src_file}.")
            shutil.copy2(src_file, dest_file)

# Example usage
# source_folder = './landmark_dataset3'
# destination_folder = './landmark_dataset'
# merge_folders(source_folder, destination_folder)



# ========================== DATASET LOADING ============================


char2int = {
            "a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8, "j":9, "k":10, "l":11, "m":12,
            "n":13, "o":14, "p":15, "q":16, "r":17, "s":18, "t":19, "u":20, "v":21, "w":22, "x":23, "y":24, 
            "z":25, "blank":26, "del":27, "space":28
            }

def load_dataset(dataset_path):
    items = []
    for root, dirs, files in os.walk(dataset_path, topdown=True):
        for file in files:
            data = np.load(os.path.join(root, file))
            if data.shape[0] != 21: 
                continue
            items.append((data,char2int[root.split(os.sep)[-1].lower()]))
            # items.append((root.split(os.sep)[-1], file))
    return items


# ========================== TRAIN MODEL ============================

def train_batch(model,batch,optimizer):
    model.train()
    points, classes = batch
    preds = model(points)
    optimizer.zero_grad()
    loss, acc = model.loss_fn(preds, classes)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

@torch.no_grad()
def validate_batch(model, data):
    model.eval()
    points, classes= data
    preds = model(points)
    loss, acc = model.loss_fn(preds, classes)

    return loss.item(), acc.item()


def train_model(model,epochs,train_dataloader, val_dataloader,folder_name,learn_rate=0.0001):
    save_folder = "./saved_models"
    os.makedirs(os.path.join(save_folder, folder_name), exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    loss_train_all = []
    acc_train_all = []
    loss_val_all = []
    acc_val_all = []
    counter = 0
    count = 0
    for epoch in range(epochs):
        print(f'Current epoch:{epoch}')
        loss_ep = []
        acc_ep = []
        for batch in train_dataloader:
            counter += 1
            loss, acc = train_batch(model, batch, optimizer)
            loss_ep.append(loss)
            acc_ep.append(acc)
        loss_train_all.append(np.mean(loss_ep))
        acc_train_all.append(np.mean(acc_ep))
        loss_ep = []
        acc_ep = []

        for batch in val_dataloader:
            loss, acc = validate_batch(model, batch)
            acc_ep.append(acc)
            loss_ep.append(loss)
        loss_val_all.append(np.mean(loss_ep))
        acc_val_all.append(np.mean(acc_ep))
        val_loss = np.mean(loss_ep)
        print(f'Loss train {loss_train_all[-1]} Loss val {loss_val_all[-1]} Acc train {acc_train_all[-1]} Acc val {acc_val_all[-1]}')
        
        count += 1
        # if count % 15 == 0:
        torch.save(model, os.path.join(os.path.join(save_folder, folder_name), f'model_{count}.pth'))
    return loss_train_all, acc_train_all, loss_val_all, acc_val_all


#  ========================== TEST MODEL ============================

@torch.no_grad()
def predict(model, img):
    model.eval()
    points_raw = get_hand_points(img)
    try:
        points = points_raw.copy()
        min_x = np.min(points_raw[:, 0])
        max_x = np.max(points_raw[:, 0])
        min_y = np.min(points_raw[:, 1])
        max_y = np.max(points_raw[:, 1])
        for i in range(len(points_raw)):
            points[i][0] = (points[i][0] - min_x) / (max_x - min_x)
            points[i][1] = (points[i][1] - min_y) / (max_y - min_y)
    except:
        return None, None

    pointst = torch.tensor([points]).float().to(device)
    label = model(pointst)
    label = label.detach().cpu().numpy()
    label = np.argmax(label)
    label = list(char2int.keys())[list(char2int.values()).index(label)]

    return label, points_raw

def calculate_accuracy(actuals, predicteds):
    if len(actuals) != len(predicteds):
        return 0
    count = 0
    for i in range(len(actuals)):
        if actuals[i] == predicteds[i]:
            count += 1
    return count / len(actuals)

def predict_images(model_path,test_path, missclassified_path):
    model = torch.load(os.path.join(model_path), map_location=torch.device("cpu"))
    # model = torch.load(os.path.join(model_path, model_name))

    os.makedirs(missclassified_path, exist_ok=True)
    shutil.rmtree(missclassified_path)

    actuals = []
    predicteds = []
    ss = set()
    signs = list(char2int.keys())
    wrongs = {}
    undetectable = {}
    errored = []
    count = 0
    for root, dirs, files in os.walk(test_path):
        gt = root.split(os.sep)[-1].lower()
        if len(gt) == 0:
            continue
        print(f'Current sign:{gt}')
        for file in files:
            if gt in signs:
                ss.add(gt)
                try:
                    img = cv2.imread(os.path.join(root, file))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img.flags.writeable = False
                    predicted_label, points = predict(model, img)
                    if predicted_label is not None:
                        predicteds.append(predicted_label)
                        actuals.append(gt)
                        if predicted_label != gt:
                            if predicted_label not in wrongs:
                                wrongs[predicted_label] = 1
                            else:
                                wrongs[predicted_label] = wrongs[predicted_label] + 1
                            os.makedirs(os.path.join(missclassified_path, predicted_label), exist_ok=True)
                            cv2.imwrite(os.path.join(missclassified_path, predicted_label, gt + '_' + file), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    else:
                        if gt not in undetectable:
                            undetectable[gt] = 1
                        else:
                            undetectable[gt] = undetectable[gt] + 1           
                except:
                    errored.append(os.path.join(root, file))
            count += 1
    
    return actuals, predicteds, count, undetectable, wrongs, errored
    

