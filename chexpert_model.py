import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt

def preprocess(images, labels):
  return tf.keras.applications.inception_v3.preprocess_input(images), labels

# Load images
chexpert_images_raw = tf.keras.utils.image_dataset_from_directory(directory = "nih3/train", image_size= (299, 299),shuffle=True, batch_size = 64 )
chexpert_images = chexpert_images_raw.map(preprocess)

# Define structure with random initialized weights
chex_model_no_top = InceptionV3(input_shape=(299,299,3), include_top= False, weights=None, pooling="avg")
chex_model = tf.keras.models.Sequential([
    chex_model_no_top,
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#chex_model = tf.keras.saving.load_model("best_3_chex_0001_random_wd_005_b_64.h5")

#Validation images
chex_images_test_raw = tf.keras.utils.image_dataset_from_directory(directory = "nih3/test", image_size= (299, 299),shuffle=True, batch_size = 64 )
chex_images_test = chex_images_test_raw.map(preprocess)

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=5,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)

mod_check = tf.keras.callbacks.ModelCheckpoint('best_4_chex_0001_random_wd_005_b_64.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# Train model
loss = tf.keras.losses.BinaryCrossentropy()
lr = 0.0001
print("best_4_chex_0001_random_wd_005_b_64.h5")
print("batches 64")
opt =  tf.keras.optimizers.Adam(learning_rate=lr, weight_decay = 0.005)
print("learning rate: ", lr)
chex_model.compile(optimizer=opt, loss=loss, metrics = ['acc'])
chex_model.fit(chexpert_images, epochs = 5, validation_data = chex_images_test, callbacks= [early_stop, mod_check])

#chex_model.layers[0].save_weights("chex_0002_random_wd_0_b_64.h5")
#chex_model.save('chex_0002_random_wd_0_b_64.keras')
#chex_model.save_weights("chex_model.h5")


