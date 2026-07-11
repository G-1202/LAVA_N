import onnxruntime as rt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AEModel(nn.Module):
    def __init__(self, input_features, output_features):
        super(AEModel, self).__init__()
        # input_features should be the flattened size of the feature maps (8400*85)
        # output_features is the number of different encoding configurations (33 in your case)

        # Define the two fully connected layers
        self.fc1 = nn.Linear(input_features, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, output_features)  # Second fully connected layer

    def forward(self, imgs):
        # f = self.get_feature(img)  # Assuming f is [N, 8400, 85] where N is the batch size
        # # Flatten the feature tensor to be [N, 8400*85]
        # f = f.view(f.size(0), -1)  # Now f is [N, 8400*85]
        # # Forward pass through the fully connected layers with ReLU activation for the first layer
        # x = F.relu(self.fc1(f))  # Apply ReLU to the output of the first fully connected layer
        # x = self.fc2(x)  # The second fully connected layer outputs the final predictions
        # return x
        batch_size = imgs.size(0)
        batch_features = []
        for i in range(batch_size):
            single_img_np = imgs[i].permute(1, 2, 0).cpu().numpy()  # Shape [H, W, C]
            single_img_feature = self.get_feature(single_img_np)
            single_img_feature_tensor = torch.tensor(single_img_feature, dtype=torch.float32).to(device)
            batch_features.append(single_img_feature_tensor)
        f = torch.cat(batch_features, dim=0)
        f = f.view(batch_size, -1)  # 例如 [N, 8400*85]
        x = F.relu(self.fc1(f))
        x = self.fc2(x)
        return x


    def get_feature(self, img):
        img_after = resize_image(img, (640, 640))
        data = img2input(img_after)
        sess = rt.InferenceSession('yolov8s.onnx')  # yolov8模型onnx格式
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        feature = sess.run([label_name], {input_name: data})[0]  # 输出(8400x84, 84=80cls+4reg, 8400=3种尺度的特征图叠加)
        feature = std_output(feature)  #（8400， 85）
        return feature


def resize_image(image, size):
    ih, iw, _ = image.shape
    h, w = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
    image_back[(h-nh)//2: (h-nh)//2 + nh, (w-nw)//2:(w-nw)//2+nw , :] = image
    return image_back


def img2input(img):
    img = np.transpose(img, (2, 0, 1))
    img = img/255
    return np.expand_dims(img, axis=0).astype(np.float32)


def std_output(pred):
    """
    （1，84，8400）→（8400， 85）  85= box:4  conf:1 cls:80
    """
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    return pred