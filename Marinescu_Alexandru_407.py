import os
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from PIL import Image
from collections import Counter
import hdbscan
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers, models, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow_hub as hub
from tensorflow.keras import backend as K
from numba import cuda

physical_devices = tf.config.list_physical_devices('GPU')

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Please install GPU version of TF")

## Had to use these to clear GPU mem because after training it wasn't freeing it
# K.clear_session()
# cuda.select_device(0)  # Select your GPU
# cuda.close()

data_directory = "data/disasters_dataset"
model_saves_directory = "model-saves"
original_directory = os.path.join(data_directory, "original")
processed_directory = os.path.join(data_directory, "processed")
train_directory = os.path.join(processed_directory, 'train')
test_directory = os.path.join(processed_directory, 'test')
valid_directory = os.path.join(processed_directory, 'valid')

os.makedirs(train_directory, exist_ok=True)
os.makedirs(valid_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

classes = [folder for folder in os.listdir(original_directory) if os.path.isdir(os.path.join(original_directory, folder))]
print("Classes found:", classes)

def count_images_in_folder(folder_path):
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

initial_images_count_data = []

for class_name in classes:
    class_path = os.path.join(original_directory, class_name)
    image_count = count_images_in_folder(class_path)
    initial_images_count_data.append({'Class': class_name, 'Image Count': image_count})
class_image_statistic = pd.DataFrame(initial_images_count_data)

plt.figure(figsize=(10, 6))
plt.bar(class_image_statistic['Class'], class_image_statistic['Image Count'], color='skyblue')
plt.title('Number of Images per Class', fontsize=16)
plt.xlabel('Class', fontsize=14)
plt.ylabel('Image Count', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.show()

def get_image_sizes(directory_params):
    widths, heights = [], []
    for class_name in os.listdir(directory_params):
        class_path = os.path.join(directory_params, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        widths.append(width)
                        heights.append(height)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
    return widths, heights

widths, heights = get_image_sizes(original_directory)

plt.figure(figsize=(10, 6))
plt.scatter(widths, heights, alpha=0.5, color='purple')
plt.title('Image Sizes Scatter Plot', fontsize=16)
plt.xlabel('Width (pixels)', fontsize=14)
plt.ylabel('Height (pixels)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

## Commented these for hardware limitations issues (433, 382) was too much
# INPUT_SIZE = (int(np.mean(widths)), int(np.mean(heights)))
# INPUT_SHAPE = (int(np.mean(widths)), int(np.mean(heights)), 3)
INPUT_SIZE = (200, 200)
INPUT_SHAPE = (200, 200, 3)
print(f"Input shape should be: {INPUT_SHAPE}")


def visualize_images_from_each_class(directory_params, classes_params, image_size_params=(150, 150)):
    num_classes = len(classes_params)
    fig, axes = plt.subplots(1, num_classes, figsize=(30, 10))
    for idx, cls in enumerate(classes_params):
        class_path = os.path.join(directory_params, cls)
        image_name = os.listdir(class_path)[0]
        image_path = os.path.join(class_path, image_name)
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(image_size_params)
        axes[idx].imshow(img_resized)
        axes[idx].set_title(cls)
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()

visualize_images_from_each_class(original_directory, classes)

CLASSES_TO_KEEP = ["drought", "earthquake", "land_slide", "urban_fire", "water_disaster", "human_damage", "infrastructure"]
COMMON_IMAGE_COUNT = class_image_statistic[class_image_statistic['Class'].isin(CLASSES_TO_KEEP)]['Image Count'].min()
SPLIT_RATIOS = {'train': 0.7, 'valid': 0.2, 'test': 0.1}
print(f"Common image count should be: {COMMON_IMAGE_COUNT}")

def preprocess_and_split_data(original_directory_params, processed_directory_params, classes_to_keep_params, input_size_params, split_ratios_params, common_image_count_params):
    for cls in classes_to_keep_params:
        class_path = os.path.join(original_directory_params, cls)
        if not os.path.exists(class_path):
            print(f"Class {cls} does not exist in the dataset.")
            continue
        images = [img for img in os.listdir(class_path) if img.endswith(('.jpg', '.png'))]
        random.shuffle(images)
        selected_images = images[:common_image_count_params]
        train, temp = train_test_split(selected_images, test_size=1 - split_ratios_params['train'], random_state=42)
        valid, test = train_test_split(temp, test_size=split_ratios_params['test'] / (split_ratios_params['valid'] + split_ratios_params['test']), random_state=42)
        splits = {'train': train, 'valid': valid, 'test': test}
        for split_name, split_images in splits.items():
            split_dir = os.path.join(processed_directory_params, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img_name in split_images:
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_resized = img.resize(input_size_params)
                    save_path = os.path.join(split_dir, img_name)
                    img_resized.save(save_path)
                except Exception as e:
                    print(f"Error processing image {img_name}: {e}")

preprocess_and_split_data(original_directory, processed_directory, CLASSES_TO_KEEP, INPUT_SIZE, SPLIT_RATIOS, COMMON_IMAGE_COUNT)

def plot_split_statistics(train_directory_params, valid_directory_params, test_directory_params, classes_params):
    def count_images_by_class(directory):
        class_counts = {}
        for cls in classes:
            class_path = os.path.join(directory, cls)
            if os.path.exists(class_path):
                class_counts[cls] = len(os.listdir(class_path))
            else:
                class_counts[cls] = 0
        return class_counts
    train_counts = count_images_by_class(train_directory_params)
    valid_counts = count_images_by_class(valid_directory_params)
    test_counts = count_images_by_class(test_directory_params)
    split_stats = pd.DataFrame({
        'Train': train_counts,
        'Valid': valid_counts,
        'Test': test_counts
    }).T
    split_stats = split_stats[classes_params]
    split_stats.plot(kind='bar', figsize=(12, 6), width=0.8)
    plt.title('Number of Images per Class in Preprocess Splits', fontsize=14)
    plt.xlabel('Split', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Classes', loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()

plot_split_statistics(train_directory, valid_directory, test_directory, CLASSES_TO_KEEP)

SUPERVISED_CNN_BATCH_SIZE = 32
SUPERVISED_CNN_IMAGE_SIZE = INPUT_SIZE
tf.random.set_seed(42)

supervised_cnn_training_dataset = image_dataset_from_directory(
    train_directory,
    batch_size=SUPERVISED_CNN_BATCH_SIZE,
    image_size=SUPERVISED_CNN_IMAGE_SIZE,
    label_mode="int",
    shuffle=True,
)
supervised_cnn_validation_dataset = image_dataset_from_directory(
    valid_directory,
    batch_size=SUPERVISED_CNN_BATCH_SIZE,
    image_size=SUPERVISED_CNN_IMAGE_SIZE,
    label_mode="int",
    shuffle=True
)
supervised_cnn_testing_dataset = image_dataset_from_directory(
    test_directory,
    batch_size=SUPERVISED_CNN_BATCH_SIZE,
    image_size=SUPERVISED_CNN_IMAGE_SIZE,
    label_mode="int",
    shuffle=True
)

def visualize_random_image(dataset_params):
    image_batch, label_batch = next(iter(dataset_params))
    random_index = np.random.randint(0, image_batch.shape[0])
    random_image = image_batch[random_index].numpy().astype("uint8")
    random_label = label_batch[random_index].numpy()
    class_label = CLASSES_TO_KEEP[int(np.argmax(random_label))]
    plt.imshow(random_image)
    plt.title(f"Label: {class_label}")
    plt.axis('off')
    plt.show()

visualize_random_image(supervised_cnn_training_dataset)

supervised_cnn_normalization_layer = layers.Rescaling(1./255)
supervised_cnn_training_dataset = supervised_cnn_training_dataset.map(lambda x, y: (supervised_cnn_normalization_layer(x), y))
supervised_cnn_validation_dataset = supervised_cnn_validation_dataset.map(lambda x, y: (supervised_cnn_normalization_layer(x), y))
supervised_cnn_testing_dataset = supervised_cnn_testing_dataset.map(lambda x, y: (supervised_cnn_normalization_layer(x), y))


def build_supervised_cnn_model(input_size_params, num_classes_params):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_size_params[0], input_size_params[1], 3)))
    model.add(layers.Conv2D(32, (2, 2), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (2, 2), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Reshape((1, -1)))
    model.add(layers.LSTM(256, activation='tanh', return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes_params, activation='softmax'))
    return model


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=5000,
    decay_rate=0.9,
    staircase=True
)

supervised_cnn_model = build_supervised_cnn_model(input_size_params=INPUT_SIZE, num_classes_params=len(CLASSES_TO_KEEP))
supervised_cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
supervised_cnn_model_history = supervised_cnn_model.fit(
    supervised_cnn_training_dataset,
    validation_data=supervised_cnn_validation_dataset,
    epochs=15,
    batch_size=SUPERVISED_CNN_BATCH_SIZE,
    verbose=1
)

supervised_cnn_model_train_accuracy = supervised_cnn_model_history.history['accuracy']
supervised_cnn_model_val_accuracy = supervised_cnn_model_history.history['val_accuracy']
supervised_cnn_model_train_loss = supervised_cnn_model_history.history['loss']
supervised_cnn_model_val_loss = supervised_cnn_model_history.history['val_loss']
supervised_cnn_model_evaluation_dataframe = pd.DataFrame({
    'epoch': range(1, len(supervised_cnn_model_train_accuracy) + 1),
    'train_accuracy': supervised_cnn_model_train_accuracy,
    'val_accuracy': supervised_cnn_model_val_accuracy,
    'train_loss': supervised_cnn_model_train_loss,
    'val_loss': supervised_cnn_model_val_loss
})

def plot_cnn_metrics(metrics_dataframe_params, title_params):
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_dataframe_params['epoch'], metrics_dataframe_params['train_accuracy'], label='Train Accuracy', marker='o', color='blue')
    plt.plot(metrics_dataframe_params['epoch'], metrics_dataframe_params['val_accuracy'], label='Validation Accuracy', marker='o', color='orange')

    plt.plot(metrics_dataframe_params['epoch'], metrics_dataframe_params['train_loss'], label='Train Loss', marker='x', linestyle='--',
             color='green')
    plt.plot(metrics_dataframe_params['epoch'], metrics_dataframe_params['val_loss'], label='Validation Loss', marker='x', linestyle='--',
             color='red')
    plt.title(title_params, fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Metrics', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_cnn_metrics(supervised_cnn_model_evaluation_dataframe, "Model Evaluation Metrics")

supervised_cnn_model_test_loss, supervised_cnn_model_test_accuracy = supervised_cnn_model.evaluate(supervised_cnn_testing_dataset, verbose=1)
print(f"Test Loss: {supervised_cnn_model_test_loss}")
print(f"Test Accuracy: {supervised_cnn_model_test_accuracy}")

def load_images_and_labels(directory_params, classes_params, input_size_params):
    images = []
    labels = []
    for class_label in classes_params:
        class_dir = os.path.join(directory_params, class_label)
        if os.path.exists(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                img = load_img(file_path, target_size=input_size_params)
                img_array = img_to_array(img) / 255.0
                images.append(img_array.flatten())
                labels.append(class_label)
    return np.array(images), np.array(labels)

supervised_ml_training_data, supervised_ml_training_labels = load_images_and_labels(train_directory, CLASSES_TO_KEEP, INPUT_SIZE)
supervised_ml_validation_data, supervised_ml_validation_labels = load_images_and_labels(valid_directory, CLASSES_TO_KEEP, INPUT_SIZE)
supervised_ml_testing_data, supervised_ml_testing_labels = load_images_and_labels(test_directory, CLASSES_TO_KEEP, INPUT_SIZE)

label_encoder = LabelEncoder()
supervised_ml_training_labels_encoded = label_encoder.fit_transform(supervised_ml_training_labels)
supervised_ml_validation_labels_encoded = label_encoder.transform(supervised_ml_validation_labels)
supervised_ml_testing_labels_encoded = label_encoder.transform(supervised_ml_testing_labels)

svm = SVC()
svc_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.01, 0.001]
}
svc_grid_search = GridSearchCV(estimator=svm, param_grid=svc_param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
svc_grid_search.fit(supervised_ml_training_data, supervised_ml_training_labels_encoded)
svc_best_params = svc_grid_search.best_params_
svc_best_score = svc_grid_search.best_score_
print(f"Best Parameters: {svc_best_params}")
print(f"Best Cross-Validation Accuracy: {svc_best_score:.4f}")

best_svc_model = svc_grid_search.best_estimator_
svc_testing_predictions = best_svc_model.predict(supervised_ml_testing_data)
print("Testing Accuracy: {:.2f}%".format(accuracy_score(supervised_ml_testing_labels_encoded, svc_testing_predictions) * 100))
print("\nClassification Report:\n")
print(classification_report(supervised_ml_testing_labels_encoded, svc_testing_predictions, target_names=CLASSES_TO_KEEP))

svc_testing_confusion_matrix = confusion_matrix(supervised_ml_testing_labels_encoded, svc_testing_predictions)
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=svc_testing_confusion_matrix, display_labels=CLASSES_TO_KEEP)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for SVM Classifier using Testing Data")
plt.xticks(rotation=45)
plt.show()

random_forest = RandomForestClassifier(random_state=42)
random_forest_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
random_forest_grid_search = GridSearchCV(
    estimator=random_forest,
    param_grid=random_forest_param_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1
)
random_forest_grid_search.fit(supervised_ml_training_data, supervised_ml_training_labels_encoded)
print("Best Parameters:", random_forest_grid_search.best_params_)

random_forest_best_score = random_forest_grid_search.best_score_
print(f"Best Cross-Validation Accuracy: {random_forest_best_score:.4f}")

best_random_forest_model = random_forest_grid_search.best_estimator_
random_forest_testing_predictions = best_random_forest_model.predict(supervised_ml_testing_data)
print("Testing Accuracy: {:.2f}%".format(accuracy_score(supervised_ml_testing_labels_encoded, random_forest_testing_predictions) * 100))
print("\nClassification Report:\n")
print(classification_report(supervised_ml_testing_labels_encoded, random_forest_testing_predictions, target_names=CLASSES_TO_KEEP))

random_forest_confusion_matrix = confusion_matrix(supervised_ml_testing_labels_encoded, random_forest_testing_predictions)
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=random_forest_confusion_matrix, display_labels=CLASSES_TO_KEEP)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for Random Forest Classifier using Testing Data")
plt.xticks(rotation=45)
plt.show()

training_directories = [train_directory, valid_directory]
unsupervised_training_images = []
unsupervised_training_labels = []
unsupervised_training_paths = []
for directory in training_directories:
    for class_folder in CLASSES_TO_KEEP:
        folder_path = os.path.join(directory, class_folder)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            try:
                img = load_img(image_path, target_size=INPUT_SIZE)
                img_array = img_to_array(img)
                img_array = img_to_array(img) / 255.0
                unsupervised_training_images.append(img_array)
                unsupervised_training_labels.append(class_folder)
                unsupervised_training_paths.append(image_path)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
unsupervised_training_images_labels = np.array(unsupervised_training_labels)
unsupervised_training_images_labels_encoded = label_encoder.fit_transform(unsupervised_training_images_labels)
unsupervised_training_images = np.array(unsupervised_training_images)

unsupervised_testing_images = []
unsupervised_testing_labels = []
unsupervised_testing_paths = []
for class_folder in CLASSES_TO_KEEP:
    folder_path = os.path.join(test_directory, class_folder)
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        try:
            img = load_img(image_path, target_size=INPUT_SIZE)
            img_array = img_to_array(img)
            unsupervised_testing_images.append(img_array)
            unsupervised_testing_labels.append(class_folder)
            unsupervised_testing_paths.append(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
unsupervised_testing_images_labels = np.array(unsupervised_testing_labels)
unsupervised_testing_images_labels_encoded = label_encoder.fit_transform(unsupervised_testing_images_labels)
unsupervised_testing_images = np.array(unsupervised_testing_images)

custom_input_layer = layers.Input(shape=INPUT_SHAPE)
custom_x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(custom_input_layer)
custom_x = layers.MaxPooling2D((2, 2))(custom_x)
custom_x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(custom_x)
custom_x = layers.MaxPooling2D((2, 2))(custom_x)
custom_x = layers.Flatten()(custom_x)
custom_bottleneck = layers.Dense(256, activation='relu')(custom_x)
custom_x = layers.Dense(50 * 50 * 64, activation='relu')(custom_bottleneck)
custom_x = layers.Reshape((50, 50, 64))(custom_x)
custom_x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(custom_x)
custom_x = layers.UpSampling2D((2, 2))(custom_x)
custom_x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(custom_x)
custom_x = layers.UpSampling2D((2, 2))(custom_x)
custom_output_layer = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(custom_x)
custom_autoencoder = Model(custom_input_layer, custom_output_layer)
custom_encoder = Model(custom_input_layer, custom_bottleneck)
custom_autoencoder.compile(optimizer='adam', loss='mse')
custom_autoencoder.fit(unsupervised_training_images, unsupervised_training_images, epochs=10, batch_size=32)

custom_cnn_training_embeddings = custom_encoder.predict(unsupervised_training_images)
custom_cnn_testing_embeddings = custom_encoder.predict(unsupervised_testing_images)
print(f"Shape of training embeddings: {custom_cnn_training_embeddings.shape}")
print(f"Shape of testing embeddings: {custom_cnn_testing_embeddings.shape}")

unsupervised_training_images_flattened = unsupervised_training_images.reshape(unsupervised_training_images.shape[0], -1)
unsupervised_testing_images_flattened = unsupervised_testing_images.reshape(unsupervised_testing_images.shape[0], -1)
pca = PCA(n_components=256)
training_embeddings_pca = pca.fit_transform(unsupervised_training_images_flattened)
testing_embeddings_pca = pca.transform(unsupervised_testing_images_flattened)
print(f"Shape of training embeddings: {training_embeddings_pca.shape}")
print(f"Shape of testing embeddings: {testing_embeddings_pca.shape}")

def get_resnet_simclr(hidden_1, hidden_2, hidden_3):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    base_model.trainable = True
    inputs = layers.Input(INPUT_SHAPE)
    h = base_model(inputs, training=True)
    h = layers.GlobalAveragePooling2D()(h)
    projection_1 = layers.Dense(hidden_1)(h)
    projection_1 = layers.Activation("relu")(projection_1)
    projection_2 = layers.Dense(hidden_2)(projection_1)
    projection_2 = layers.Activation("relu")(projection_2)
    projection_3 = layers.Dense(hidden_3)(projection_2)
    resnet_simclr = Model(inputs, projection_3)
    return resnet_simclr
resnet_simclr_model = get_resnet_simclr(512, 256, 128)

def preprocess_images_resnet_simclr(images_params):
    images_resized = np.array([img_to_array(load_img(image_path, target_size=INPUT_SIZE)) for image_path in images_params])
    images_resized = images_resized / 255.0
    return images_resized
def extract_embeddings_resnet_simclr(model_params, images_params):
    images_preprocessed = preprocess_images_resnet_simclr(images_params)
    embeddings = model_params.predict(images_preprocessed)
    return embeddings
resnet_simclr_model_training_embeddings = extract_embeddings_resnet_simclr(resnet_simclr_model, unsupervised_training_paths)
resnet_simclr_model_testing_embeddings = extract_embeddings_resnet_simclr(resnet_simclr_model, unsupervised_testing_paths)
print(f"Shape of training embeddings: {resnet_simclr_model_training_embeddings.shape}")
print(f"Shape of testing embeddings: {resnet_simclr_model_testing_embeddings.shape}")


def evaluate_hdbscan(min_cluster_size_params, min_samples_params, embeddings_params):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_params,
                                min_samples=min_samples_params,
                                metric='euclidean')
    clusterer.fit(embeddings_params)
    cluster_labels = clusterer.labels_
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    if num_clusters > 1:
        silhouette = silhouette_score(embeddings_params, cluster_labels)
    else:
        silhouette = -1
    return num_clusters, silhouette

hdbscan_param_grid = {
    'min_cluster_size': [2, 3, 5, 10, 20, 50],
    'min_samples': [1, 2, 5, 10, 15]
}

hdbscan_cnn_results = []
for min_cluster_size in hdbscan_param_grid['min_cluster_size']:
    for min_samples in hdbscan_param_grid['min_samples']:
        num_clusters, silhouette = evaluate_hdbscan(min_cluster_size, min_samples, custom_cnn_training_embeddings)
        hdbscan_cnn_results.append((min_cluster_size, min_samples, num_clusters, silhouette))
hdbscan_cnn_embedding_method_training_dataframe = pd.DataFrame(hdbscan_cnn_results, columns=['min_cluster_size', 'min_samples', 'num_clusters', 'silhouette_score'])

hdbscan_cnn_best_result = hdbscan_cnn_embedding_method_training_dataframe.sort_values(by='silhouette_score', ascending=False).iloc[0]
print("\nBest Clustering Result:")
print(f"min_cluster_size: {hdbscan_cnn_best_result['min_cluster_size']}")
print(f"min_samples: {hdbscan_cnn_best_result['min_samples']}")
print(f"Number of clusters: {hdbscan_cnn_best_result['num_clusters']}")
print(f"Silhouette Score: {hdbscan_cnn_best_result['silhouette_score']}")

hdbscan_cnn_real_cluster_count_params = {
    'min_cluster_size': 5,
    'min_samples': 2
}
hdbscan_cnn_cluster_labels = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2).fit_predict(custom_cnn_training_embeddings)
hdbscan_cnn_unique_labels, hdbscan_cnn_counts = np.unique(hdbscan_cnn_cluster_labels, return_counts=True)
hdbscan_cnn_non_noise_labels = hdbscan_cnn_unique_labels[hdbscan_cnn_unique_labels != -1]
hdbscan_cnn_non_noise_counts = hdbscan_cnn_counts[hdbscan_cnn_unique_labels != -1]
plt.figure(figsize=(10, 6))
plt.bar(hdbscan_cnn_non_noise_labels.astype(str), hdbscan_cnn_non_noise_counts, color='lightgreen')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Samples')
plt.title('Sample Distribution per Cluster (Excluding Noise)')
plt.xticks(rotation=45)
plt.show()

tsne_model = TSNE(n_components=2, random_state=42, metric='euclidean')
hdbscan_cnn_tsne_embeddings = tsne_model.fit_transform(custom_cnn_training_embeddings)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(hdbscan_cnn_tsne_embeddings[:, 0], hdbscan_cnn_tsne_embeddings[:, 1], c=hdbscan_cnn_cluster_labels, cmap='viridis', s=50)
plt.colorbar(scatter)
plt.title("Clusters visualized in 2D space using t-SNE")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()

hdbscan_pca_results = []
for min_cluster_size in hdbscan_param_grid['min_cluster_size']:
    for min_samples in hdbscan_param_grid['min_samples']:
        num_clusters, silhouette = evaluate_hdbscan(min_cluster_size, min_samples, training_embeddings_pca)
        hdbscan_pca_results.append((min_cluster_size, min_samples, num_clusters, silhouette))
hdbscan_pca_embedding_method_training_dataframe = pd.DataFrame(hdbscan_pca_results, columns=['min_cluster_size', 'min_samples', 'num_clusters', 'silhouette_score'])

hdbscan_pca_best_result = hdbscan_pca_embedding_method_training_dataframe.sort_values(by='silhouette_score', ascending=False).iloc[0]
print("\nBest Clustering Result:")
print(f"min_cluster_size: {hdbscan_pca_best_result['min_cluster_size']}")
print(f"min_samples: {hdbscan_pca_best_result['min_samples']}")
print(f"Number of clusters: {hdbscan_pca_best_result['num_clusters']}")
print(f"Silhouette Score: {hdbscan_pca_best_result['silhouette_score']}")

hdbscan_pca_closest_cluster_count_params = {
    'min_cluster_size': 3,
    'min_samples': 2
}

hdbscan_pca_cluster_labels = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2).fit_predict(training_embeddings_pca)
hdbscan_pca_unique_labels, hdbscan_pca_counts = np.unique(hdbscan_pca_cluster_labels, return_counts=True)
hdbscan_pca_non_noise_labels = hdbscan_pca_unique_labels[hdbscan_pca_unique_labels != -1]
hdbscan_pca_non_noise_counts = hdbscan_pca_counts[hdbscan_pca_unique_labels != -1]
plt.figure(figsize=(10, 6))
plt.bar(hdbscan_pca_non_noise_labels.astype(str), hdbscan_pca_non_noise_counts, color='lightgreen')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Samples')
plt.title('Sample Distribution per Cluster (Excluding Noise)')
plt.xticks(rotation=45)
plt.show()

tsne_model = TSNE(n_components=2, random_state=42, metric='euclidean')
tsne_embeddings = tsne_model.fit_transform(training_embeddings_pca)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=hdbscan_pca_cluster_labels, cmap='viridis', s=50)
plt.colorbar(scatter)
plt.title("Clusters visualized in 2D space using t-SNE")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()

hdbscan_pretrained_results = []
for min_cluster_size in hdbscan_param_grid['min_cluster_size']:
    for min_samples in hdbscan_param_grid['min_samples']:
        num_clusters, silhouette = evaluate_hdbscan(min_cluster_size, min_samples, resnet_simclr_model_training_embeddings)
        hdbscan_pretrained_results.append((min_cluster_size, min_samples, num_clusters, silhouette))
hdbscan_pretrained_embedding_method_training_dataframe = pd.DataFrame(hdbscan_pretrained_results, columns=['min_cluster_size', 'min_samples', 'num_clusters', 'silhouette_score'])

hdbscan_pretrained_best_result = hdbscan_pretrained_embedding_method_training_dataframe.sort_values(by='silhouette_score', ascending=False).iloc[0]
print("\nBest Clustering Result:")
print(f"min_cluster_size: {hdbscan_pretrained_best_result['min_cluster_size']}")
print(f"min_samples: {hdbscan_pretrained_best_result['min_samples']}")
print(f"Number of clusters: {hdbscan_pretrained_best_result['num_clusters']}")
print(f"Silhouette Score: {hdbscan_pretrained_best_result['silhouette_score']}")

hdbscan_pretrained_closest_cluster_count_params = {
    'min_cluster_size': 5,
    'min_samples': 1
}

hdbscan_pretrained_cluster_labels = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1).fit_predict(resnet_simclr_model_training_embeddings)
hdbscan_pretrained_unique_labels, hdbscan_pretrained_counts = np.unique(hdbscan_pretrained_cluster_labels, return_counts=True)
hdbscan_pretrained_non_noise_labels = hdbscan_pretrained_unique_labels[hdbscan_pretrained_unique_labels != -1]
hdbscan_pretrained_non_noise_counts = hdbscan_pretrained_counts[hdbscan_pretrained_unique_labels != -1]
plt.figure(figsize=(10, 6))
plt.bar(hdbscan_pretrained_non_noise_labels.astype(str), hdbscan_pretrained_non_noise_counts, color='lightgreen')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Samples')
plt.title('Sample Distribution per Cluster (Excluding Noise)')
plt.xticks(rotation=45)
plt.show()

tsne_model = TSNE(n_components=2, random_state=42, metric='euclidean')
tsne_embeddings = tsne_model.fit_transform(resnet_simclr_model_training_embeddings)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=hdbscan_pretrained_cluster_labels, cmap='viridis', s=50)
plt.colorbar(scatter)
plt.title("Clusters visualized in 2D space using t-SNE")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()


def evaluate_agglomerative(n_clusters_params, linkage_params, embeddings_params):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters_params,
                                        linkage=linkage_params,
                                        metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings_params)
    num_clusters = len(set(cluster_labels))
    if num_clusters > 1:
        silhouette = silhouette_score(embeddings_params, cluster_labels)
    else:
        silhouette = -1
    return silhouette


agglomerative_param_grid = {
    'n_clusters': [2, 3, 4, 5, 6, 7, 8],
    'linkage': ['ward', 'complete', 'average', 'single']
}

agglomerative_cnn_results = []
for n_clusters in agglomerative_param_grid['n_clusters']:
    for linkage in agglomerative_param_grid['linkage']:
        silhouette = evaluate_agglomerative(n_clusters, linkage, custom_cnn_training_embeddings)
        agglomerative_cnn_results.append((n_clusters, linkage, silhouette))
agglomerative_cnn_embedding_method_training_dataframe = pd.DataFrame(agglomerative_cnn_results, columns=['n_clusters', 'linkage', 'silhouette_score'])

agglomerative_cnn_best_result = agglomerative_cnn_embedding_method_training_dataframe.sort_values(by='silhouette_score', ascending=False).iloc[0]
print("\nBest Clustering Result:")
print(f"n_clusters: {agglomerative_cnn_best_result['n_clusters']}")
print(f"linkage: {agglomerative_cnn_best_result['linkage']}")
print(f"Silhouette Score: {agglomerative_cnn_best_result['silhouette_score']}")

agglomerative_all_real_cluster_count_params = {
    'n_clusters': 7,
    'linkage': 'average'
}

agglomerative_cnn_cluster_labels = AgglomerativeClustering(n_clusters=7, linkage='average', metric='euclidean').fit_predict(custom_cnn_training_embeddings)
agglomerative_cnn_unique_labels, agglomerative_cnn_counts = np.unique(agglomerative_cnn_cluster_labels, return_counts=True)
plt.figure(figsize=(10, 6))
plt.bar(agglomerative_cnn_unique_labels.astype(str), agglomerative_cnn_counts, color='lightgreen')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Samples')
plt.title('Sample Distribution per Cluster (Excluding Noise)')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=agglomerative_cnn_embedding_method_training_dataframe, x="n_clusters", y="silhouette_score", hue="linkage", marker="o")
plt.title("Silhouette Score by Number of Clusters and Linkage Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.legend(title="Linkage Method")
plt.grid(True)
plt.tight_layout()
plt.show()

tsne_model = TSNE(n_components=2, random_state=42, metric='euclidean')
tsne_embeddings = tsne_model.fit_transform(custom_cnn_training_embeddings)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=agglomerative_cnn_cluster_labels, cmap='viridis', s=50)
plt.colorbar(scatter)
plt.title("Clusters visualized in 2D space using t-SNE")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()

agglomerative_pca_results = []
for n_clusters in agglomerative_param_grid['n_clusters']:
    for linkage in agglomerative_param_grid['linkage']:
        silhouette = evaluate_agglomerative(n_clusters, linkage, training_embeddings_pca)
        agglomerative_pca_results.append((n_clusters, linkage, silhouette))
agglomerative_pca_embedding_method_training_dataframe = pd.DataFrame(agglomerative_pca_results, columns=['n_clusters', 'linkage', 'silhouette_score'])

agglomerative_pca_best_result = agglomerative_pca_embedding_method_training_dataframe.sort_values(by='silhouette_score', ascending=False).iloc[0]
print("\nBest Clustering Result:")
print(f"n_clusters: {agglomerative_pca_best_result['n_clusters']}")
print(f"linkage: {agglomerative_pca_best_result['linkage']}")
print(f"Silhouette Score: {agglomerative_pca_best_result['silhouette_score']}")

agglomerative_pca_cluster_labels = AgglomerativeClustering(n_clusters=7, linkage='average', metric='euclidean').fit_predict(training_embeddings_pca)
agglomerative_pca_unique_labels, agglomerative_pca_counts = np.unique(agglomerative_pca_cluster_labels, return_counts=True)
plt.figure(figsize=(10, 6))
plt.bar(agglomerative_pca_unique_labels.astype(str), agglomerative_pca_counts, color='lightgreen')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Samples')
plt.title('Sample Distribution per Cluster (Excluding Noise)')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=agglomerative_pca_embedding_method_training_dataframe, x="n_clusters", y="silhouette_score", hue="linkage", marker="o")
plt.title("Silhouette Score by Number of Clusters and Linkage Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.legend(title="Linkage Method")
plt.grid(True)
plt.tight_layout()
plt.show()

tsne_model = TSNE(n_components=2, random_state=42, metric='euclidean')
tsne_embeddings = tsne_model.fit_transform(training_embeddings_pca)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=agglomerative_pca_cluster_labels, cmap='viridis', s=50)
plt.colorbar(scatter)
plt.title("Clusters visualized in 2D space using t-SNE")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()

agglomerative_pretrained_results = []
for n_clusters in agglomerative_param_grid['n_clusters']:
    for linkage in agglomerative_param_grid['linkage']:
        silhouette = evaluate_agglomerative(n_clusters, linkage, resnet_simclr_model_training_embeddings)
        agglomerative_pretrained_results.append((n_clusters, linkage, silhouette))
agglomerative_pretrained_embedding_method_training_dataframe = pd.DataFrame(agglomerative_pretrained_results, columns=['n_clusters', 'linkage', 'silhouette_score'])

agglomerative_pretrained_best_result = agglomerative_pretrained_embedding_method_training_dataframe.sort_values(by='silhouette_score', ascending=False).iloc[0]
print("\nBest Clustering Result:")
print(f"n_clusters: {agglomerative_pretrained_best_result['n_clusters']}")
print(f"linkage: {agglomerative_pretrained_best_result['linkage']}")
print(f"Silhouette Score: {agglomerative_pretrained_best_result['silhouette_score']}")

agglomerative_pretrained_cluster_labels = AgglomerativeClustering(n_clusters=7, linkage='average', metric='euclidean').fit_predict(resnet_simclr_model_training_embeddings)
agglomerative_pretrained_unique_labels, agglomerative_pretrained_counts = np.unique(agglomerative_pretrained_cluster_labels, return_counts=True)
plt.figure(figsize=(10, 6))
plt.bar(agglomerative_pretrained_unique_labels.astype(str), agglomerative_pretrained_counts, color='lightgreen')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Samples')
plt.title('Sample Distribution per Cluster (Excluding Noise)')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=agglomerative_pretrained_embedding_method_training_dataframe, x="n_clusters", y="silhouette_score", hue="linkage", marker="o")
plt.title("Silhouette Score by Number of Clusters and Linkage Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.legend(title="Linkage Method")
plt.grid(True)
plt.tight_layout()
plt.show()

tsne_model = TSNE(n_components=2, random_state=42, metric='euclidean')
tsne_embeddings = tsne_model.fit_transform(resnet_simclr_model_training_embeddings)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=agglomerative_pretrained_cluster_labels, cmap='viridis', s=50)
plt.colorbar(scatter)
plt.title("Clusters visualized in 2D space using t-SNE")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()


def evaluate_random_clustering(num_clusters_params, embeddings_params):
    random_labels = np.random.randint(0, num_clusters_params, size=len(embeddings_params))
    if num_clusters_params > 1:
        silhouette = silhouette_score(embeddings_params, random_labels)
    else:
        silhouette = -1
    return num_clusters_params, silhouette

cnn_random_cluster_params = range(2, 11)
random_results = []
for num_clusters in cnn_random_cluster_params:
    num_clusters, silhouette = evaluate_random_clustering(num_clusters, custom_cnn_training_embeddings)
    random_results.append((num_clusters, silhouette))
random_method_training_dataframe = pd.DataFrame(random_results, columns=['num_clusters', 'silhouette_score'])

random_best_result = random_method_training_dataframe.sort_values(by='silhouette_score').iloc[0]
print("\nBest Clustering Result:")
print(f"num_clusters: {random_best_result['num_clusters']}")
print(f"Silhouette Score: {random_best_result['silhouette_score']}")

random_real_cluster_count_params = {
    'num_clusters': 7
}
random_labels = np.random.randint(0, 7, size=len(custom_cnn_training_embeddings))
random_unique_labels, random_counts = np.unique(random_labels, return_counts=True)
plt.figure(figsize=(10, 6))
plt.bar(random_unique_labels.astype(str), random_counts, color='lightgreen')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Samples')
plt.title('Sample Distribution per Cluster (Random Choice)')
plt.xticks(rotation=45)
plt.show()

tsne_model = TSNE(n_components=2, random_state=42, metric='euclidean')
tsne_embeddings = tsne_model.fit_transform(custom_cnn_training_embeddings)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=random_labels, cmap='viridis', s=50)
plt.colorbar(scatter)
plt.title("Clusters visualized in 2D space using t-SNE")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()

testing_embeddings_methods = {
    "custom_cnn": custom_cnn_testing_embeddings,
    "pca": testing_embeddings_pca,
    "simclr": resnet_simclr_model_testing_embeddings
}
hdbscan_params = {
    "custom_cnn": hdbscan_cnn_real_cluster_count_params,
    "pca": hdbscan_pca_closest_cluster_count_params,
    "simclr": hdbscan_pretrained_closest_cluster_count_params
}
agglomerative_params = {
    "custom_cnn": agglomerative_all_real_cluster_count_params,
    "pca": agglomerative_all_real_cluster_count_params,
    "simclr": agglomerative_all_real_cluster_count_params
}

def evaluate_clustering_on_testing(cluster_labels_params, embeddings_params, true_labels_params):
    ari = adjusted_rand_score(true_labels_params, cluster_labels_params)
    nmi = normalized_mutual_info_score(true_labels_params, cluster_labels_params)
    if len(set(cluster_labels_params)) > 1:
        silhouette = silhouette_score(embeddings_params, cluster_labels_params)
    else:
        silhouette = -1
    return ari, nmi, silhouette


final_testing_results = []
for embedding_name, embeddings in testing_embeddings_methods.items():
    hdbscan_clusterer = hdbscan.HDBSCAN(**hdbscan_params[embedding_name], metric='euclidean')
    hdbscan_cluster_labels = hdbscan_clusterer.fit_predict(embeddings)
    ari, nmi, silhouette = evaluate_clustering_on_testing(hdbscan_cluster_labels, embeddings,
                                                          unsupervised_testing_images_labels_encoded)
    final_testing_results.append(["HDBSCAN", embedding_name, ari, nmi, silhouette])
    agglomerative_clusterer = AgglomerativeClustering(**agglomerative_params[embedding_name], metric='euclidean')
    agglomerative_cluster_labels = agglomerative_clusterer.fit_predict(embeddings)
    ari, nmi, silhouette = evaluate_clustering_on_testing(agglomerative_cluster_labels, embeddings,
                                                          unsupervised_testing_images_labels_encoded)
    final_testing_results.append(["Agglomerative", embedding_name, ari, nmi, silhouette])
    random_labels = np.random.randint(0, len(CLASSES_TO_KEEP), size=len(embeddings))
    ari, nmi, silhouette = evaluate_clustering_on_testing(random_labels, embeddings, unsupervised_testing_images_labels_encoded)
    final_testing_results.append(["Random Choice", embedding_name, ari, nmi, silhouette])
print(final_testing_results)

final_results_dataframe = pd.DataFrame(final_testing_results, columns=["Method", "Embedding", "ARI", "NMI", "Silhouette"])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.barplot(data=final_results_dataframe, x="Embedding", y="ARI", hue="Method", palette="Set2", ax=axes[0])
axes[0].set_title("Adjusted Rand Index (ARI) by Method and Embedding")
axes[0].set_ylabel("ARI")
sns.barplot(data=final_results_dataframe, x="Embedding", y="NMI", hue="Method", palette="Set2", ax=axes[1])
axes[1].set_title("Normalized Mutual Information (NMI) by Method and Embedding")
axes[1].set_ylabel("NMI")
sns.barplot(data=final_results_dataframe, x="Embedding", y="Silhouette", hue="Method", palette="Set2", ax=axes[2])
axes[2].set_title("Silhouette Score by Method and Embedding")
axes[2].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

cluster_to_class_map = {}
for cluster in set(agglomerative_pretrained_cluster_labels):
    cluster_indices = [i for i, label in enumerate(agglomerative_pretrained_cluster_labels) if label == cluster]
    cluster_true_labels = [unsupervised_training_images_labels_encoded[i] for i in cluster_indices]
    majority_class = Counter(cluster_true_labels).most_common(1)[0][0]
    cluster_to_class_map[cluster] = majority_class
print(cluster_to_class_map)