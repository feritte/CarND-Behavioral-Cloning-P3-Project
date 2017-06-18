#**Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Project3.PNG "Model Code"
[image2]: ./examples/P3TrainingValidationOutput.PNG "Training and Validation accuracy output"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipngy containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model consists of a 3 convolutional layers and 2 fully connected layers. The convolutional layers have a SAME padding & elu activation and the fully connected layers have elu activation as well.The data is normalized in the model using a Keras lambda layer. The model summary can be found inside the ipynb file. The model summary is given as:

* cropping2d_1 (Cropping2D) layer which has (None, 65, 320, 3) outputshape and has 0 parameters.
* lambda_1 (Lambda) layer which has (None, 65, 320, 3) output shape and has 0 parameters
* convolution2d_1 (Convolution2D)  (None, 17, 80, 16) and has 3088 parameters.
* elu_1 (ELU) which has (None, 17, 80, 16) output shape. 
* convolution2d_2 (Convolution2D) which has (None, 9, 40, 32) output shape and has 12832 parameters.
* elu_2 (ELU) which has (None, 9, 40, 32).
* convolution2d_3 (Convolution2D) which has (None, 5, 20, 64) output shape and has 51264       
* flatten_1 (Flatten) which has (None, 6400).
* elu_3 (ELU) which has (None, 6400) output shape. 
* dense_1 (Dense) which has (None, 512) and has 3277312 paramaters.
* dropout_1 (Dropout) which has  (None, 512).
* elu_4 (ELU) which has (None, 512).
* dense_2 (Dense) which has (None, 50) and has 25650 parameters.
* elu_5 (ELU) which has (None, 50).
* dense_3 (Dense) which has (None, 1) and has 51 parameters. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting at the first fully connected layer (a dropout of 0.5). 

I trained with 5 epochs. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

I used the training data provided by Udacity. . I used a combination of center lane driving, recovering from the left and right sides of the road.


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to add 3 convolutional layers. 
For the first convolutional layer, I used 16 filters, 8 X 8 kernals, 4 X 4 strides, and same padding and elu activation For the second convolutional layer, I used 32 filters, 5 X 5 kernals, 2 X 2 strides, and same padding and elu activation For the third convolutional layer, I used 62 filters, 5 X 5 kernals, 2 X 2 strides, and same padding and elu activation. To combat the overfitting, I modified the model so that there is a drobout.

I split my data into a training set and validation set.



The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is given as following : 

![alt text][image1]



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the dataset provided by Udacity.

As it is adviced in the introduction of the project, I cropped the images. I realized that not all parts of the image are necessary for driving/steering. Then normalizing the data for a cleaner convergenceis applied. Lastly, I shuffled the data. 95% of the data went into the validation set.

To avoid any offfset, I took all three camera images, and added a correction of 0.2.

The validation set was to diagnose my model, particularly tell whether it had high bias (underfitting) or high variance (overfitting)

I found that 5 epochs was ideal. Here is a visualization of the training and validation accuracy output 

![alt text][image2]

Run1.mp4 is showing the simulator output of the trained model in autonomous mode. 


