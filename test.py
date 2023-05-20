import numpy as np
import argparse
from path import Path

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from score_utils import mean_score, std_score
from PIL import Image
import torchvision.models
import torchvision.transforms as transforms


def nima(img_path):
  if img_path == 0:
    return np.nan
  else:
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('./mobilenet_weights.h5')

    score_list = []

    

    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    scores = model.predict(x, batch_size=1, verbose=0)[0]

    mean = mean_score(scores)
    std = std_score(scores)

    file_name = Path(img_path).name.lower()
    score_list.append((file_name, mean))
    return mean
  
def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    Transform = transforms.Compose([
            transforms.Resize([224,224]),      
            transforms.ToTensor(),
            ])
    image = Transform(image)   
    image = image.unsqueeze(0)
    return image.to(device)

def predict_ii(image, model):
    if image == 0:
      return np.nan
    else:
      image = Image.open(image)
      image = prepare_image(image)
      with torch.no_grad():
          preds = model(image)
      return preds.item()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet50()
# model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
model.fc = torch.nn.Linear(in_features=2048, out_features=1)
model.load_state_dict(torch.load('/content/model-resnet50.pth', map_location=device)) 
model.eval().to(device)
df_img['iipa'] = predict_ii(x, model)