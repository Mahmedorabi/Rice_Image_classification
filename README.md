# Rice Image Classification Dataset
### This repository contains a dataset of rice images for image classification tasks. The dataset can be used to train and evaluate machine learning models for various applications, such as:

- Rice variety classification (e.g., Basmati, Jasmine, Arborio)
Dataset Description:

- The dataset is comprised of images of rice grains and/or rice plants. Each image is labeled with a corresponding class (e.g., rice variety, quality grade). The specific details of the dataset will be further elaborated upon in the dataset_info.txt file.

## Data Format:

- The images are stored in a folder structure. Each subfolder represents a specific class, and the images within that folder belong to that class. The file format of the images will be specified in the dataset_info.txt file (e.g., JPEG, PNG).


## Explore the dataset:

- The images are located in the data folder.
Refer to dataset_info.txt for details about the class labels, image format, and any additional information.
Preprocess the data (if necessary):

- You may need to preprocess the data depending on your specific needs and machine learning model. This might involve  resizing images, converting color spaces, or data augmentation.

## Train your model:

- Use the provided dataset to train your image classification model. Split the data into training, validation, and test sets for optimal model performance evaluation.

## Evaluate your model:

- Evaluate your trained model's performance on the test set using relevant metrics like accuracy

## Contribution:

- We welcome contributions to this dataset. If you have rice images that you would like to share, please feel free to submit a pull request with the following:

#### Labeled rice images following the specified folder structure.
#### An update to dataset_info.txt to reflect the addition of your data.


# Rice Image Classification

This project is focused on classifying different types of rice grains using a convolutional neural network (CNN). The model is trained and evaluated on a dataset of rice images.

## Project Overview

The goal of this project is to build and evaluate a deep learning model capable of accurately classifying images of different types of rice. The project includes data preprocessing, model building, training, evaluation, and prediction.



## Notebook Structure

The Jupyter Notebook `Rice Image Classification.ipynb` contains the following sections:

1. **Importing Libraries:**
    Necessary libraries such as TensorFlow, NumPy, and Matplotlib are imported.

    ```python
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    import numpy as np
    import matplotlib.pyplot as plt
    ```


2. **Data Loading and Preprocessing:**
    The dataset is loaded, and preprocessing steps such as resizing and normalization are applied.

    ```python
    # Example: Loading data
    train_data = ...
    test_data = ...
    ```
3. **Visualize Data**
   ![Rice](https://github.com/Mahmedorabi/Rice_Image_classification/assets/105740465/23e94d72-555b-4c0e-a39b-dacda66b006e)

3. **Model Building:**
    A Convolutional Neural Network (CNN) model is built using Keras.

    ```python
    model = models.Sequential()
    # input layer --> faltten layer

   model.add(Flatten())

   # Hidden layer 
   model.add(Dense(265,activation='relu'))
   model.add(Dense(128,activation='relu'))
   model.add(Dense(64,activation='relu'))
   model.add(Dense(64,activation='relu'))

   # output layer 
   model.add(Dense(5,activation='softmax'))
    ```

4. **Model Compilation and Training:**
    The model is compiled with appropriate loss function, optimizer, and metrics, and then trained on the training dataset.

    ```python
    # model compile
   model.compile(optimizer='adam',
   loss='categorical_crossentropy',metrics=['accuracy'])
    ```

5. **Model Evaluation:**
    The trained model is evaluated on the test dataset to determine its accuracy and loss.

    ```python
    # Evaluate Test set
      test_loss,test_acc=model.evaluate(test_data)
    ```

6. **Predictions:**
    Predictions are made on new data, and the results are processed and displayed.

    ```python
    y_pred = model.predict(test_data)
    ypred = np.argmax(y_pred, axis=1)
    ```

## Results

- **Training Accuracy:** 97.05%
- **Test Accuracy:** 97.17%

## Conclusion

The CNN model demonstrates high accuracy in classifying different types of rice grains, showing its effectiveness for this task.


