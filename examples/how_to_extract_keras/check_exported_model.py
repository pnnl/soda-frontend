from keras.models import model_from_json

json_file = open('../resnet-50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../resnet-50.h5")
# print("Loaded model from disk")

print(model.to_json(indent=4))

model.summary()

for lay in model.layers:
    if lay.name != "conv2d":
        continue
    print(lay.name)
    print(lay.get_weights())
