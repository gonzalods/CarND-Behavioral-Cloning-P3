# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia-model.png "Model Visualization"
[image2]: ./examples/center.jpg "Center lane"
[image3]: ./examples/recover1.jpg "Recovery Image"
[image4]: ./examples/recover2.jpg "Recovery Image"
[image5]: ./examples/recover3.jpg "Recovery Image"
[image6]: ./examples/left.jpg "Left Camera"
[image7]: ./examples/center.jpg "Center Camera"
[image8]: ./examples/right.jpg "Right Camera"
[image9]: ./examples/center.jpg "Normal Image"
[image10]: ./examples/flipping.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing a test lap on track one

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with a combinatio of 5x5 and 3x3 filter sizes and depths between 24 and 64 and following by three full-conected layers.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Data is also cropped 70 pixels at the top and 25 pixels at the bottom of the images using Keras cropping layer.

#### 2. Attempts to reduce overfitting in the model

The model contains some dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer and the learning rate was tuned to a value of 0.0001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving laps in right and opposite direction, and laps recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train the model incrementally.

My first step was to use a convolution neural network model similar to the nvidia's model with a few training data, to the birge of the track one. Then I noticed that OpenCV reads the images in BGR scale and that drive.py reads them in RGB. So that, I converted the data to the RGB scale. This increased the performance remarkably. Afterwards, I increased the amount of data supplied to the model.

To combat the overfitting I introduced dropout layers in the model, adjusting their number. Tambien ajusté la tasa de aprendizaje en el optimizador.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model has the same arquitecture of the nvidia's arquitecture that appears in its [paper](https://arxiv.org/pdf/1604.07316v1.pdf).
It consists of a convolution neural network with the first 3 convolutional layers with 5x5 filter sizes and depths of 24, 36 and 48. Next, two more convolutional layers
with 3x3 filter size and a detph of 64 each. Finally, there are 3 full-connected layer with 100, 50 and 10 neurons each layer.

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one fast lap on track one using center lane driving and a second lap driving more slowly in the curves. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to the center of the lane. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

I used the images of the three cameras, correcting the angle of the central camera in an amount of plus-minus 0.255 (left-reight).

![alt text][image6]
![alt text][image7]
![alt text][image8]

To augment the data set, I also flipped images and angles thinking that this would generalize more the model. For example, here is an image that has then been flipped:

![alt text][image9]
![alt text][image10]


After the collection process, I had X number of data points.

I then randomly shuffled the data set and put 20% of the data into a validation set. 

I finally preprocessed this data by normalizing them and cropping them within Kera´s model.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 5 since they hardly improved performance with more if not at all.

I used an adam optimizer with a learning rate of 0.0001.
