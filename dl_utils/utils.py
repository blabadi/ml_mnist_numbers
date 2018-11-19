from keras.models import model_from_yaml
import os

def saveModel(model, directory, name):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    os.makedirs(os.path.dirname(directory), exist_ok=True)
    with open(directory + name + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(directory + name + ".h5")
    print("Saved model to disk")


def loadModel(path, name):
    # load YAML and create model
    yaml_file = open(path + name + '.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(path + name + ".h5")
    print("Loaded model from disk")
    return loaded_model