import numpy as np
from keras.applications.inception_v3 import InceptionV3
import os
import tensorflow as tf
from scipy.linalg import fractional_matrix_power
import itertools
import json


def calculate_fid(feat1, feat2):
    '''
    calculate fid between two sets of images
    return: fid score
    '''
    # calculate mean and covariance of features
    mu1, sigma1 = np.mean(feat1, axis=0), np.cov(feat1, rowvar=False) 
    mu2, sigma2 = np.mean(feat2, axis=0), np.cov(feat2, rowvar=False)
    # sum squared difference between means
    diff = np.sum((mu1 - mu2)**2)
 
    # calculate geometric mean of covariance matrices
    covmean = fractional_matrix_power(sigma1.dot(sigma2), 0.5)
    # check for imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    fid = diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def preprocess(images, labels):
  return tf.keras.applications.inception_v3.preprocess_input(images), labels

def get_features(images, model_weights, class_idx):
    '''
    Load Inceptionv3 model and get feature vectors of images
    return: feature vectors
    '''

    model = InceptionV3(input_shape=(299,299,3), include_top= False, weights=model_weights, pooling="avg")
    all_features = model.predict(images)

    features = {}
    for idx in class_idx:
       features[idx] = all_features[class_idx[idx]]
    
    return features


def fid_experiment(image_path, models):
    images_raw = tf.keras.utils.image_dataset_from_directory(directory = image_path, image_size= (299, 299),shuffle=False)
    class_names = images_raw.class_names
    images_all = images_raw.map(preprocess)
    # Get labels of all images
    labels = []
    for im, la in images_all:
        labels.append(la.numpy())
    labels = np.concatenate(labels)
    num_labels = len(set(labels))

    # Create dictionary of indexs of each class
    class_idx = {}
    smallest_class = 10*10**5
    for i in range(num_labels):
        class_idx[i] = labels==i
        if np.sum(class_idx[i]) < smallest_class:
            smallest_class = np.sum(class_idx[i])
    print(smallest_class)

    for model in models:
        features = get_features(images_all, model, class_idx)
        if len(features[3]) >= 2*smallest_class: #Check if number of normal samples is greater than two times the smallest class
            for infiltration in np.arange(0,smallest_class, int(smallest_class/4)):
                a = features[3][:smallest_class]
                b = np.concatenate((features[3][samllest_class : 2*smallest_class - infiltration], features[0][:infiltration]), axis = 0)
                fid = calculate_fid(a, b)
                print(f"FID between {class_names[3]} and {class_names[3]} for model:{model} with infiltration: {infiltration/smallest_class*100} % from {class_names[0]}")
                print(f"Fid: {fid}")
            for infiltration in np.arange(0,smallest_class, int(smallest_class/4)):
                a = features[3][:smallest_class]
                b = np.concatenate((features[3][samllest_class : 2*smallest_class - infiltration], features[1][:infiltration]), axis = 0)
                fid = calculate_fid(a, b)
                print(f"FID between {class_names[3]} and {class_names[3]} for model:{model} with infiltration: {infiltration/smallest_class*100} % from {class_names[1]} ")
                print(f"Fid: {fid}")

        else:
            smallest_class = int(len(features[3])/2) - 1
            for infiltration in np.arange(0,smallest_class, int(smallest_class/4)):
                a = features[3][:smallest_class]
                b = np.concatenate((features[3][samllest_class : 2*smallest_class - infiltration], features[0][:infiltration]), axis = 0)
                fid = calculate_fid(a, b)
                print(f"FID between {class_names[3]} and {class_names[3]} for model:{model} with infiltration: {infiltration/smallest_class*100} % from {class_names[0]}")
                print(f"Fid: {fid}")
            for infiltration in np.arange(0,smallest_class, int(smallest_class/4)):
                a = features[3][:smallest_class]
                b = np.concatenate((features[3][samllest_class : 2*smallest_class - infiltration], features[1][:infiltration]), axis = 0)
                fid = calculate_fid(a, b)
                print(f"FID between {class_names[3]} and {class_names[3]} for model:{model} with infiltration: {infiltration/smallest_class*100} % from {class_names[1]} ")
                print(f"Fid: {fid}")
        print(smallest_class)
        with open(f'heart_lung_features_{model}.json', 'w') as file:
            json.dump({'heart_features': features[0].tolist(), 'lung_features': features[1].tolist(), 'lung_heart_features': features[2].tolist(), 'normal_features': features[3].tolist()}, file)



if __name__ == "__main__":
    models = ["no_top_best_3_chex_0001_random_wd_005_b_64.h5", "imagenet", "RadImageNet-InceptionV3_notop.h5"]
    image_path = "heart_lung_images"
    fid_experiment(image_path, models)