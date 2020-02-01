# Class-activation-Map

## Introduction

* Ability to localize objects in the convolutional layers 
* This ability is lost when fully-connected layers are used for classification.

![image](https://user-images.githubusercontent.com/55071900/73592594-452ef300-4526-11ea-92a3-2ff2c94f7f93.png)

![image](https://user-images.githubusercontent.com/55071900/73592602-60016780-4526-11ea-84fd-cf4341b41696.png)

# Fully-convolutional neural networks are used to achieve localize objects

# Class Activation Mapping

* A class activation map (CAM) lets us see which regions in the image were relevant to which class.​
* CAM draws the heatmap of the network that shows the activated region. 

* Just before the final output layer  
  * performed global average pooling (GAP) on the convolutional feature maps ​
  * Using those as features for a fully-connected layer that produces the desired output​

![image](https://user-images.githubusercontent.com/55071900/73592675-eddd5280-4526-11ea-9f46-3640ce3e5909.png)

# Global average pooling

* In contrast, GAP keep the information about the every pixel.
* Uses global average pooling which acts as a structural regularizer, preventing over-fitting during training.

# Model Used VGG-16 to implement CAM
* Used vgg16 because it preformed better in classifying disaster images  

# Performed CAM on Disaster dataset 

![fires1200_Predicted_fires_prob_0 24977395_](https://user-images.githubusercontent.com/55071900/73592757-ba4ef800-4527-11ea-88ca-533a2be026e2.png)

![fires1200](https://user-images.githubusercontent.com/55071900/73592762-c5098d00-4527-11ea-87bd-3685367198f1.png)



  
