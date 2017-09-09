# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is similar to the Navidia model. It consists of several convolution neural layers and dense layers.
Here is a description of each layer：

![model_visualization.jpg](http://upload-images.jianshu.io/upload_images/2528310-e9a4da88e282c342.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

The model includes RELU layers to introduce nonlinearity (code line 63-69), and the data is normalized in the model using a Keras lambda layer (code line 61). 

#### 2. Attempts to reduce overfitting in the model

The model doesn't contain dropout layers. We use earlystopping to avoid overfitting, and the model is only trained for 1-2 epochs,  and the model runs well. If we run it at least 5 epochs, the model will be overfitting (the performance is very bad on track 1).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 78).
Why the adam optimizer performs well without manually tuning parameters?

#### 4. Appropriate training data

I didn't use the provided images, and collected only 25000+ images by myself. All the images are used for training. I used a combination of center lane driving, recovering from the left and right sides of the road. The left and right images of the camera have a steering correction 0.35. The codes are as follows:
```
# center: num==0, left: num==1, right: num == 2
if num % 3 == 0:
image_dir = all_data[0][num]
angle = y[num]
elif num % 3 == 1:
image_dir = all_data[1][num]
angle = y[num] + 0.35
else:
image_dir = all_data[2][num]
angle = y[num] - 0.35
```

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the nividia model. I thought this model might be appropriate because the model is deep enough. Deep convolutional neural network can learn high level features which may be very helpful for behavioral cloning.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collect more data and increase the steering correction ration, which greatly improves the performance.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 60-74) consisted of a convolution neural network with the following layers and layer sizes .

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
```

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
![model.png](http://upload-images.jianshu.io/upload_images/2528310-ccc57bd9c3e88c90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:


![center lane driving](http://upload-images.jianshu.io/upload_images/2528310-f8034952c89755b7.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to center. These images show what a recovery looks like
![righit side of the road](http://upload-images.jianshu.io/upload_images/2528310-92fac6319847224f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) :

![Start to turn left a little bit](http://upload-images.jianshu.io/upload_images/2528310-e53f3e6da0a2d00e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![Go back to center](http://upload-images.jianshu.io/upload_images/2528310-7fd9348cbab8d4ad.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


After the collection process, I had 25710 number of data points. I then preprocessed this data by normalizing  the pixels in images with a lambda layer and removing the top part of the images, which are irrelevant to self-driving.

```
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
```


I used this training data for training the model. The ideal number of epochs was 2 as evidenced by the performance on track 1. I used an adam optimizer so that manually training the learning rate wasn't necessary.
