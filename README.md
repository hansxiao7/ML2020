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
## HW2 - Image Classification with CNN
- Use Convolutional Neural Network for image classification. 11 different classes of images are given.
- Kaggle link: [here](https://www.kaggle.com/c/ml2020spring-hw3/overview);
- TensorFlow is used for image processing and model training;
  - Note: TensorFlow V2.3 is used with command image_dataset_from_directory. The command has a bug ([link](https://github.com/tensorflow/tensorflow/issues/44752)), which cannot load label lists. Images should be manually seperated by classes into different subdirectories. 
- Five layers of convolutional layers + max pooling are used, with 3 layers of regular layers after flatten;
- After 30 epoches, training accuracy is 96.9%, and validation error is 43.5%. The reason why validation error is low needs further study.
<div align=center><img width="800" height="600" src="https://github.com/hansxiao7/ML2020/blob/main/HW3/Accuracy.jpg"/></div>
