# ML2020
This is to store all homework assigned in ML2020 course.  
Course website is (http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html), including all projects and homework info.

## HW1 - Linear Regreesion
- Use linear regression and gradient descent to estimate PM2.5 values;
- Kaggle link: [here](https://www.kaggle.com/c/ml2020spring-hw1/overview);
- HW1 code link: [here](https://github.com/hansxiao7/ML2020/tree/main/HW1);
- Three modeling schemes are given, and the method description can be seen [here](https://github.com/hansxiao7/ML2020/blob/main/HW1/HW1%20Method%20Description.pdf):
  - [Method 1](https://github.com/hansxiao7/ML2020/tree/main/HW1/Trial%201)
  - [Method 2](https://github.com/hansxiao7/ML2020/tree/main/HW1/Trial%202)
  - [Sample given by Hung-yi Lee](https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C)

## HW2 - 2-class Classification
- Use a Generative Model (Gaussian distribution) and a Discriminative Model (Logistic regression) to accomplish a 2-class classfication; 
- Kaggle link: [here](https://www.kaggle.com/c/ml2020spring-hw2/overview);
- HW2 code link: [here](https://github.com/hansxiao7/ML2020/tree/main/HW2);
  - Training cases and validation cases are chosen randomly from the original training data set
  - Generative model: [here](https://github.com/hansxiao7/ML2020/blob/main/HW2/Generative_model.py)
    - Training loss: 12.4%
    - Validation loss: 12.6%
  - Discriminative model with SGD: [here](https://github.com/hansxiao7/ML2020/blob/main/HW2/Discriminative_model.py)
    - Learning rate = 0.00001
    - Learning iterations = 1000
    - Training loss: 11.4%
    - Validation loss: 11.9%
  - For this data set, since the number of data sets is relatively large, the discriminative method yields a smaller loss for both training and validation. When the number of training data is small, the generative model may yield a better result because of the assumption of probability distribution.

## HW3 - Image Classification with CNN
- Use Convolutional Neural Network for image classification. 11 different classes of images are given.
- Kaggle link: [here](https://www.kaggle.com/c/ml2020spring-hw3/overview);
- TensorFlow is used for image processing and model training;
  - Note: TensorFlow V2.3 is used with command image_dataset_from_directory. The command has a bug ([link](https://github.com/tensorflow/tensorflow/issues/44752)), which cannot load label lists. Images should be manually seperated by classes into different subdirectories. 
- Five layers of convolutional layers + max pooling are used, with 3 layers of regular layers after flatten;
- After 30 epoches, training accuracy is 96.9%, and validation error is 43.5%. The reason why validation error is low needs further study.
<div align=center><img width="600" height="450" src="https://github.com/hansxiao7/ML2020/blob/main/HW3/Accuracy.jpg"/></div>

## HW4 - Text Classification with RNN and Semi-Supervised Learning (self learning)
- Use RNN and self learning for text classification. Output is a binary class, with 1 as positive sentence and 0 for negative sentence;
- Kaggle link: [here](https://www.kaggle.com/c/ml2020spring-hw4);
- 70% of labelled texts are used as the training data, and the remaining 30% labelled texts are used for validation;
- TensorFlow is used for model building and training. Bidirectional LSTM layer is applied as the RNN structure;
- For self learning, 0.8 is chosen as the threshold to get pseudo-labels for unlabelled data. For each unlabelled data, if output is larger than 0.8, the data will be labelled as 1; if the output is smaller than 0.2, the data will be labelled as 0. Remaining data will not be labelled/used for training. 
- In each training, 1,000 unlabelled data will be predicted. After considering possibility threshold, newly labelled data are added into original training data set for training. In this project, 500~600 cases are added to the training set in each epoch.
- After 10 epoches of learning without using unlabelled data, the accuracy is 97.3% for training data, and is 74.7% for validation data;
- After 10 epoches of self-learning with considering 1,000 unlabelled data, the accuracy is 97.5% for the training data, and is 74.5% for validation data.
- CoLab link: [here](https://colab.research.google.com/drive/1pODwohWg5TmFkb5tB-96fvUYoq9R-K3c?usp=sharing)

## HW5 - Explaniable ML
- Use the trained CNN in HW3 to build saliency maps and filter visualization;
- Saliency map: [here](https://github.com/hansxiao7/ML2020/tree/main/HW5/Task%201%20-%20Saliency%20Map)
- Saliency maps of two pictures are shown as examples. The first picture is a 'bread', and the second picture is a 'dairy';
<img src="https://github.com/hansxiao7/ML2020/blob/main/HW5/Task%201%20-%20Saliency%20Map/0_0.jpg" width="200"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW5/Task%201%20-%20Saliency%20Map/saliency_map_0.jpeg" width="200"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW5/Task%201%20-%20Saliency%20Map/1_29.jpg" width="200"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW5/Task%201%20-%20Saliency%20Map/saliency_map_1.jpeg" width="200"/>
- Filter visualization: [here](https://github.com/hansxiao7/ML2020/tree/main/HW5/Task%202%20-%20Filter%20Visualization)
