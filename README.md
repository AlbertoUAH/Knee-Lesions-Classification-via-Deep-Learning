# Knee-Lesions-Classification-via-Deep-Learning
Knee Lesions Classification via Deep Learning Techniques (MRNet model)

# Summary
The main purpose of this project is training and deploy a Deep Learning model for knee lesions detection:

* Anterior Cruciate Ligament tear or ACL Tear: __319 positive cases__
* Meniscus Tear: __508 positive cases__
* Another abnormalities or "Abnormal": __1104 positive cases__

# Data
Training and validation data were extracted from [Stanford Machine Learning Group Repository](https://stanfordmlgroup.github.io/competitions/mrnet/):
* Train data: contains 1130 .npy files
* Validation data: contains 120 .npy files

# Model architecture
In a nutshell, the architecture consists of three-pretrained AlexNet models using PyTorch (only the convolutional part is selected). Convolutional outputs are concatenated into a Fully Conected layer (FC) to get the probability of each lesion (sigmoid function):

![Triple MRNet Model architecture](./media/architecture_diagram.png)


# Training and validation results

| Train acc. | Validation acc. | Abnormal sens. | Abnormal spec. | ACL sens. | ACL spec. | Meniscus sens. | Meniscus spec. |
|------------|-----------------|----------------|----------------|-----------|-----------|----------------|----------------|
| 0.7991     | 0.7972          | 0.9053         | 0.72           | 0.8939    | 0.7037    | 0.6923         | 0.7353         |

# GitHub content
Table of Contents
=================
   * ./src/main.py: python main file for webapp execution
   * ./src/utils.py: python file which contains application corpus such as constants, functions and MRNet class (deep learning architecture)
   * ./src/train.ipynb: python notebook for model training (trained on Google Colab - GPU)
   
    * learning_rate: 1e-05
    * data augmentation (imgaug library)
    * pos_weights on Loss function
    
   * ./data: folder with few samples of MRI validation data

# Web application
The final web application for model deployment was built using streamlit library (Python). To deploy the app, you must run the following line in your command line or cmd:

```
streamlit run main.py
```

This web application shows not only labels probabilities (Abnormal, ACL and Meniscus tear), but also image heatmap via GradCAM algorithm:

![Web application gif sample](./media/webapp_gif.gif)

# References
[Stanford Machine Learning Repository](https://stanfordmlgroup.github.io/competitions/mrnet/)

[Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet / Plos Medicine Journal](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699)

[Stanford MRNet Challenge: Classifying Knee MRIs](https://learnopencv.com/stanford-mrnet-challenge-classifying-knee-mris/)

[Triple MRNet Architecture sample](https://github.com/yashbhalgat/MRNet-Competition)

[Image augmentation library](https://github.com/aleju/imgaug)

[Implemented Grad-CAM in PyTorch for VGG16 Network](https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82)

[GradCAM algorithm explained](https://glassboxmedicine.com/2020/05/29/grad-cam-visual-explanations-from-deep-networks/#:~:text=Grad%2DCAM%20is%20a%20form,and%20the%20parameters%20are%20fixed.)

[Streamlit library](https://streamlit.io/)


