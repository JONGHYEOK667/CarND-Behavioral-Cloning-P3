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

[image1]: ./output_fig/1.collected_data.jpg "1.collected_data"
[image2]: ./output_fig/2.Augmented_Image(Centering).jpg "Centering"
[image3]: ./output_fig/2.Augmented_Image(BiasedRecover).jpg "2.Augmented_Image(BiasedRecover)"
[image4]: ./output_fig/2.Augmented_Image(EdgeRecover).jpg "2.Augmented_Image(EdgeRecover)"
[image5]: ./output_fig/3.Augmented_data.jpg "3.Augmented_data"
[image6]: ./output_fig/4.Train_History.jpg "4.Train_History"

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

And, here is a link to my [Jonghyeok's project notebook](https://github.com/JONGHYEOK667/Udacity_SelfDrivingCar_P4/blob/main/SDC_P4.ipynb), [Jonghyeok's project python code](https://github.com/JONGHYEOK667/Udacity_SelfDrivingCar_P4/blob/main/model.py)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy  

#### 0. Architecture layer
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| cropping         		| up = 70, down = 25, left = 0, right = 0, outputs 65x320x3 							| 
| lambda     	| x: x/255-0.5 (Nomalize),  outputs 65x320x3 |
| Convolution 	      	| 24 filters, 5x5 kernel, 2x2 stride, RELU activation, outputs 33x160x24|
| Convolution      	| 36 filters, 5x5 kernel, 2x2 stride, RELU activation, outputs 17x80x36|
| Convolution	      	|48 filters, 5x5 kernel, 2x2 stride, RELU activation, outputs 9x40x48|
| Convolution      	| 64 filters, 3x3 kernel, 1x1 stride, RELU activation, outputs 9x40x64|
| Drop out      	| rate = 0.2,  outputs 9x40x64 				|
| Convolution      	| 64 filters, 3x3 kernel, 1x1 stride, RELU activation, outputs 9x40x64|
| Flatten    	| outputs 23040|
| Fully connected	      	| outputs 100 				|
| Drop out      	| rate = 0.4 				|
| Fully connected     	| outputs 50 				|
| Drop out		| rate = 0.4      									|
| Fully connected		| outputs 10      									|
| Fully connected		| outputs 1      									|
|:---------------------:|:---------------------------------------------:| 
|	Total parameters					|					2,441,019							|
|			Trainable parameters					|						2,441,019									|


#### 1. An appropriate model architecture has been employed

My model was transformed based on the model by Nvidia.  [Reference](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)   

In the pre-processing part, define the ROI area using the cropping layer. And, image array is normalized through the lambda layer   

In the feature extraction part, five convolution layers using relu activation is connected sequentially.  

Finally, in the regression part, the steering value is finally predicted using 4 fully cennected layers.  

#### 2. Attempts to reduce overfitting in the model

To prevent overfitting, two dropout layers is applied fully connected laryers and one convolution layers.   

Then, adjust the drop rate manually by monitoring trend of loss and val_loss.
 
Finally, Use Early stopping callback `(patience = 5,monitor='val_loss',mode = 'min')` 

#### 3. Appropriate training data


The overall strategy for deriving a model architecture was to drive within the lane

so, Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and reverse direction driving etc....  

As a result, I saved the training date that has a distribution of steering angle (label) as follows.

![alt text][image1]

Because of counter-clock wise test track, it can be seen  that negative steering counts is more than positive counts.   
However, through Data augmentation using Flip, it is possible to create a symatric distribution.    


Data augmentation used not only flip image but also correction image by left and right cameras.  
There are 6 types of data used in this project as follows.

![alt text][image2] 



This image is case on driving center of lane ideally. 
In addition, data was also collected when driving is biased compared to lane center,   
and when driving outside boundary of lane

![alt text][image3]
![alt text][image4]


Therefore, the labels of all train data have the following distribution.


![alt text][image5]


I used this training data for training the model. The validation set helped determine if the model was over or under fitting.  
I randomly shuffled the data set and put 30% of the data into a validation set. 
I made a generator for save memory of the machine
I choose the 'Adam' as optimizer of the network. because, Adam can adapt the moment and step size. also it is most prefered algorithm. 


Finally, the training history is as follows (solid line : loss(mse) / mae,  dashed line : val_loss(val_mse) / val_mae)

![alt text][image6]

