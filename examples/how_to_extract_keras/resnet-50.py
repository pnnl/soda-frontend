from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

from tensorflow.keras.models import model_from_json
json_string = model.to_json()
with open(r'resnet-50.json', 'w') as file:
    file.write(json_string)

# save weights
model.save_weights('resnet-50.h5')
model.summary()
