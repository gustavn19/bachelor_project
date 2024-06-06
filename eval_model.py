import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import accuracy_score
import json

def preprocess(images, labels):
  return tf.keras.applications.inception_v3.preprocess_input(images), labels
loaded_model = tf.keras.saving.load_model("best_3_chex_0001_random_wd_005_b_64.h5")
#loaded_model.layers[0].save_weights("no_top_best_chex_00001_random_wd_05_b_64.h5")
#print("New model saved")
#loaded_model = tf.keras.saving.load_model("best_chex_0001_random_wd_005_b_64.h5")
#base_model = InceptionV3(input_shape=(299,299,3), include_top= False, weights="no_top_best_chex_0001_random_wd_005_b_64.h5", pooling="avg")
#loaded_model = tf.keras.models.Sequential([
#    base_model,
#    tf.keras.layers.Dense(1, activation='sigmoid')
#])
#Load test images
chexpert_images_raw = tf.keras.utils.image_dataset_from_directory(directory = "nih3/val", image_size= (299, 299),shuffle=False)
chexpert_images = chexpert_images_raw.map(preprocess)

labels = []
    # Go through each image and extract label
for im, la in chexpert_images:
    labels.append(la.numpy())
labels = np.concatenate(labels)

predictions = loaded_model.predict(chexpert_images)
pred_labels = ((predictions > 0.5)+0).ravel()


print(accuracy_score(labels, pred_labels))

# Save the lists to a file
with open('labels_data_00001.json', 'w') as file:
    json.dump({'pred_labels': pred_labels.tolist(), 'labels': labels.tolist(), 'predictions': predictions.tolist()}, file)



