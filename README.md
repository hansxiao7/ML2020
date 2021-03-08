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
<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW5/Task%201%20-%20Saliency%20Map/0_0.jpg" width="200"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW5/Task%201%20-%20Saliency%20Map/saliency_map_0.jpeg" width="200"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW5/Task%201%20-%20Saliency%20Map/1_29.jpg" width="200"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW5/Task%201%20-%20Saliency%20Map/saliency_map_1.jpeg" width="200"/></div>

- Filter visualization: [here](https://github.com/hansxiao7/ML2020/tree/main/HW5/Task%202%20-%20Filter%20Visualization)

## HW6 - Adversarial Attack
- Use FGSM (Fast Gradient Sign Method) to generate adversarial images;
- FGSM code: [here](https://github.com/hansxiao7/ML2020/blob/main/HW6/FGSM.py)
- Pixel limit for adversarial images is set to 15 for more obvious results;
- VGG16 is used as the proxy network;
- The predicted label for Image 0 is ground bettle (49.1%) before attack. After attack, the label is cardigan (23.1%);

<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW6/data/images/000.png" width="200"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW6/data/attacked/0.png" width="200"/></div>

- The predicted label for Image 1 is vase (49.3) before attack. After attack, the label is mosque (24.4%).

<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW6/data/images/001.png" width="200"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW6/data/attacked/1.png" width="200"/></div>

## HW7 - Network Compression
- This homework includes applications of network pruning, knowledge distillation, parameter quantization, and architecture design;
- Network pruning: remove less important weights/neurons after training, then fine tune the pruned model;
- [Knowledge distillation](https://github.com/hansxiao7/ML2020/blob/main/HW7/knowledge_distillation.py): a smaller 'student' model learns 'everything' output by the 'teacher' model, not only the final output;
- Parameter quantization: use less bits or weight clustering to reduce the size of a model (e.g., change weights from float64 to int8);
- [Achitecture design](https://github.com/hansxiao7/ML2020/blob/main/HW7/depthwise_pointwise_conv.py): by adding intermediate layers, the total number of parameters in a model can be reduced (e.g., depthwise & pointwise CNN);
- In Tensorflow, [SeparableConv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SeparableConv2D) layer can be directly used for depthwise & pointwise CNN. The code in the folder is to show how to build complicated architectures in TensorFlow.

## HW8 - Seq2Seq Model for Translation
- This homework is to use different seq2seq models to translate English to Chinese;
- Colab is used to run codes with GPU, all Colab files are uploaded [here](https://github.com/hansxiao7/ML2020/blob/main/HW8/seq2seq_teacher_forcing.ipynb);
- Method 1: [Teacher-forcing seq2seq](https://colab.research.google.com/drive/1mcveqsvBMtaSQ8WNJ1C4h4o_VCuQkgmP?usp=sharing)
  - 256 LSTM units are used, with training accuray 95% after 100 epoches;
  - Some translation examples:
    - English: mary is sitting at the desk                                           
      Chinese: 瑪麗的男人正在著一本書。
    - English: i will be free in ten minutes                                          
      Chinese: 我十分鐘後有空。
    - English: i ve got no friends                                            
      Chinese: 我沒有朋友。
    - English: what time does the movie start                                           
      Chinese: 電影什麼時候開始？
  - Network structure:
  <div align=center><img width="600" height="300" src="https://github.com/hansxiao7/ML2020/blob/main/HW8/Method%201%20structure.png"/></div>
- Method 2: [Attention model - Bahdanau's additive style](https://github.com/hansxiao7/ML2020/blob/main/HW8/Attention_teacher_forcing.ipynb)
  - 256 LSTM units are used, with training accuracy 99.5% after 60 epoches;
  - Some translation examples:
    - English: he is tall                                              
      Chinese: 他很高。
    - English: hurry up and you will be in time for the bus                                      
      Chinese: 在上快點，快點！了這快點是真的。
    - English: he s always at home on sundays                                          
      Chinese: 他星期日總是在家。
    - English: i do nt know if i have the time                                        
      Chinese: 我不知道我有時間。
    - English: did he go to see mary                                           
      Chinese: 他去看見瑪麗了嗎？
  - Network structure:
  <div align=center><img width="600" height="800" src="https://github.com/hansxiao7/ML2020/blob/main/HW8/Attention_model.png"/></div>
- Method 3: Transformer
  - Reference: [here](https://www.tensorflow.org/tutorials/text/transformer)

## HW9 - Image Autoencoder with Unsupervised Learning
- This homework is to classify images with unlabelled data. Images are classfied with 'Natural View' and 'No Natural View';
- Kaggle link: [here](https://www.kaggle.com/c/ml2020spring-hw9)
- CoLab link: [here](https://github.com/hansxiao7/ML2020/blob/main/HW9/denoising_autoencoder.ipynb)
- The De-noising autoencoder-decoder model with 2 de-convolution layers (Conv2DTranspose) is used. Image original sizes (32, 32, 3) are embedded into 128-element vectors, then transformed to 2-D vectors by t-SNE;
- Original images:
<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/original_1.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/original_2.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/original_3.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/original_6.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/original_7.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/original_9.png" width="50"/></div>

- Noised images:
<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/noised_1.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/noised_2.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/noised_3.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/noised_6.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/noised_7.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/noised_9.png" width="50"/></div>

- Images output by decoders:
<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/decoded_1.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/decoded_2.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/decoded_3.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/decoded_6.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/decoded_7.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW9/output_images/decoded_9.png" width="50"/></div>

- Accuracy in validation set: 70.8%


## HW10 - Anomaly Detection
- This homework is to detect anomaly with autoencoders. After reconstruction with the trained autoencoder, if the reconstruction error is larger than a setted threshold, the data is considered as an anomaly;
- Squared error 200 is set as the threshold. After running, 62/10000 (0.62%) test data is considered as anomaly;
- Kaggle link: [here](https://www.kaggle.com/c/ml2020spring-hw10)
- CoLab link: [here](https://github.com/hansxiao7/ML2020/blob/main/HW10/autoencoder_anomaly.ipynb)
- Trained images:
<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/train_1.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/train_2.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/train_3.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/train_4.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/train_5.png" width="50"/></div>

- Image considered as normal in test data:
<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/normal_1.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/normal_2.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/normal_3.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/normal_4.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/normal_5.png" width="50"/></div>

- Image considered as anomaly in test data:
<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/anomaly_1.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/anomaly_2.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/anomaly_3.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/anomaly_4.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW10/images/anomaly_5.png" width="50"/></div>

## HW11 - GAN
- This homework is to use DCGAN (Deep Convolutional GAN) to generate anime profile photos.
- Data link: [here](https://crypko.ai/#)
- Two models were built for GAN. Model structures are the same, but the training strategies are different. Model 1 used gradients to update parameters directly, and Model 2 used tf.keras.Model.fit and tf.keras.Model.compile instead of using gradients directly;
- Model 1 CoLab link: [here](https://github.com/hansxiao7/ML2020/blob/main/HW11/GAN.ipynb)
- Model 2 CoLab link: [here](https://github.com/hansxiao7/ML2020/blob/main/HW11/GAN_2.ipynb)
- Both models were trained with 10000 photos and 50 epochs.
- Image generated by Model 1:
<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/1.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/2.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/3.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/4.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/5.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/6.png" width="50"/></div>

- Image generated by Model 2:
<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/11.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/12.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/13.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/14.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/15.png" width="50"/><img src="https://github.com/hansxiao7/ML2020/blob/main/HW11/images/16.png" width="50"/></div>

- In both models, when using generator or discriminator to predict values, **training=True** is necessary to activate BatchNormalization layers. Batch normalization layers are important for a large model like GAN. For instance:
```json
generated_images = generator(noise, training=True)
real_output = discriminator(real_images, training=True)
fake_output = discriminator(generated_images, training=True)
```

## HW12 - Transfer Learning - Domain Adverserial Neural Network
- This homework is use DaNN to train unlabelled target data with labelled source data;
- Kaggle link: [here](https://www.kaggle.com/c/ml2020spring-hw12)
- The idea of GaNN is to build a Feature Extractor, a Label Predictor, and a Domain Classifier;
<div align=center><img src="https://pabebezz.github.io/article/2d45afe4/%E5%9F%9F%E5%AF%B9%E6%8A%97%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.png" width="600"/></div>

- Label predictor: match labels for source data, and predict labels for target data;
- Feature extractor: extract image features, maximize label classification accuracy and minimize domain classification accuracy;
- Domain classifier: decide which domain is the feature coming from (from source or target). Target domain is set to 1, and source domain is set to 0;
- Target images and source images are preprocessed by PIL: [preprocess code](https://github.com/hansxiao7/ML2020/blob/main/HW12/data_extract.py)
- Becasue of limited GPU capacity, the whole network is trained with 15 epochs. The structure works according to a trail with limited data. The code is available for reference [here].(https://github.com/hansxiao7/ML2020/blob/main/HW12/DaNN.ipynb)
- The labelling accuracy for source data is 99.0% (4951/5000);
- For the first 20 target images, the predicted labels are shown in the following image. Only 35% (7/20) are predicted correctly.
<div align=center><img src="https://github.com/hansxiao7/ML2020/blob/main/HW12/pred_results.PNG" width="1000"/></div>

## HW13 - Meta Learning
- This homework is to apply meta learning. The original task is to change codes with 2nd order gradients to codes with 1st order gradient approximiation;
- The original codes are available here: [for regression](https://colab.research.google.com/drive/1MFJwRdOTefd6UOYRsNjdc7BWuB7Qe3lY), [for few-shot classification](https://colab.research.google.com/drive/1OcF5TQCCd7WNK0cbXyzYxAzWpMKW_r8B);
- A MAML code with 1st order gradient for regression can be seen [here](https://github.com/hansxiao7/ML2020/blob/main/HW13/MAML_regression.ipynb);
- Meta learning code with reptile gradient: [here](https://github.com/hansxiao7/ML2020/blob/main/HW13/reptile_regression.ipynb);
- Reference: [Paper repro: Deep Metalearning using “MAML” and “Reptile”](https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0)

## HW14 - Lifelong Learning with EWS
- This homework is to use EWS to achieve lifelong learning; 
- Task 1: digit recognition on MNIST dataset;
- Task 2: digit recognition on SVHN dataset;
- The image size for MNIST is (28, 28, 1), and the image size for SVHN is (32, 32, 3). [Data preprocessing](https://github.com/hansxiao7/ML2020/blob/main/HW14/data_transform.py) is conducted first to transfer SVHN data with the same dimension of MNIST data. An ImageNet-like network is built for these two tasks;
- Without EWS, the cross-entropy loss table for these two tasks is shown as follows. The model is firstly trained with MNIST data, then trained with SVHN data.

|   | Test on MNIST | Test on SVHN |
|     :---:      |     :---:      |     :---:      |
|Random Init.   | 2.94     | 2.51    |
| MNIST Trained   | 0.09     | 11.89   |
| SVHN Trained     | 2.43  | 0.69  |

- With EWS and learning rate for EWS = 10, the cross-entropy loss table for these two tasks is shown as follows:

|  | Test on MNIST | Test on SVHN |
|     :---:      |     :---:      |     :---:      |
|Random Init.   | 2.54    | 2.64    |
| MNIST Trained   | 0.08     | 14.82   |
| SVHN Trained     | 0.09  | 4.24  |

- With EWS and learning rate for EWS = 0.001, the cross-entropy loss table for these two tasks is shown as follows:

|   | Test on MNIST | Test on SVHN |
|     :---:      |     :---:      |     :---:      |
|Random Init.   | 2.91    | 2.65    |
| MNIST Trained   | 0.07     | 15.12   |
| SVHN Trained     | 0.41  | 0.93  |

- Codes are available [here](https://github.com/hansxiao7/ML2020/blob/main/HW14/life_long_learning.ipynb);

## HW15 - Reinforcement Learning
- Homework can be found [here](https://colab.research.google.com/drive/1Q5H0NI5b_NrT1ZUxEuFMd8ASTbow55Qj).
