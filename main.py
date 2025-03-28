import argparse
from tqdm import tqdm
import cv2
import torch
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk

import torchvision.models as models

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.file_list = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                self.file_list.append((os.path.join(class_dir, file_name), class_name))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label, 'image_path': img_path}

def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_features(dataset_path, batch, num_images):

    device = 'mps'

    model = models.resnet18(pretrained=True)
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)

    features = []
    labels = []
    image_paths = []

    for batch in tqdm(dataloader, desc='Running the model inference'):
        images = batch['image'].to(device)
        labels.extend(batch['label'])
        image_paths.extend(batch['image_path'])

        with torch.no_grad():
            output = model(images)

        features.append(output.cpu().numpy())

        if len(labels) >= num_images:
            break

    features = np.concatenate(features)
    print("Unique labels:", set(labels))
    return features[:num_images], labels[:num_images], image_paths[:num_images]

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

import numpy as np
from scipy.spatial import cKDTree

class TSNEVisualizer:
    def __init__(self, features, labels, image_paths):
        self.features = features
        self.labels = labels
        self.image_paths = image_paths
        self.colors_per_class = {
            'stamped': (0.9, 0.0, 0.0),
            'ok': (0.0, 0.9, 0.0),
            'scratch': (0.0, 0.0, 0.9),
            'tape': (0.9, 0.9, 0.0),
            'scorch': (0.9, 0.0, 0.9)
        }

        self.root = tk.Tk()
        self.root.title("t-SNE Visualization")

        self.fig = plt.figure(figsize=(20, 10))
        self.tsne_ax = self.fig.add_subplot(121)
        self.image_ax = self.fig.add_subplot(122)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.perform_tsne()
        self.plot_tsne()

        self.tree = cKDTree(np.column_stack((self.tx, self.ty)))

        self.canvas.mpl_connect('button_press_event', self.on_click)

        self.root.mainloop()

    def perform_tsne(self):
        tsne = TSNE(n_components=2).fit_transform(self.features)
        self.tx, self.ty = tsne[:, 0], tsne[:, 1]
        self.tx = scale_to_01_range(self.tx)
        self.ty = scale_to_01_range(self.ty)

    def plot_tsne(self):
        self.tsne_ax.clear()
        unique_labels = set(self.labels)

        for label in unique_labels:
            indices = [i for i, l in enumerate(self.labels) if l == label]
            current_tx = np.take(self.tx, indices)
            current_ty = np.take(self.ty, indices)
            self.tsne_ax.scatter(current_tx, current_ty, c=[self.colors_per_class[label]], label=label)

        self.tsne_ax.legend(loc='best')
        self.tsne_ax.set_title('t-SNE Visualization')
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes == self.tsne_ax:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                dist, index = self.tree.query([x, y], k=1)
                if dist < 0.01:  # Adjust this threshold as needed
                    self.show_image(index)

    def show_image(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path)
        self.image_ax.clear()
        self.image_ax.imshow(img)
        self.image_ax.axis('off')
        self.image_ax.set_title(f'Class: {self.labels[index]}')
        self.canvas.draw()

def main():
    fix_random_seeds()

    # 경로를 자신의 환경에 맞게 수정해주세요
    features, labels, image_paths = get_features("./dataset", 8, 3000)
    print(f'Features extracted, shape: {features.shape}')

    visualizer = TSNEVisualizer(features, labels, image_paths)

if __name__ == '__main__':
    main()