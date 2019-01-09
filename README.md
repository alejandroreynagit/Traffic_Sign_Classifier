# Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Writeup - Daniel Alejandro Reyna Torres

In this project, goal is to write a software pipeline to identify and recognise traffic signs.

---

## The Project

The goals / steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

## Data Set Summary & Exploration

The very fist step in every Machine Learning task is to load and understand the data. The data set corresponds to traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). I used the numpy and pandas libraries to calculate summary statistics of the traffic signs data set:


Above is an **exploratory visualization** of the training set. Summary of data is:

- Number of training examples = 34799
- Number of testing examples = 12630
- Number of validating examples = 4410
- Image data shape = (32, 32, 3)
- Number of classes = 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the amount of samples for each traffic sign is distributed. 

![dataset_dist]

If we take a closer loog on the data set we can see the following: 

![class_freq]

Traffic signs with most samples:
- Speed limit (50km/h) - 2010 samples
- Speed limit (30km/h) - 1980 samples
- Yield - 1920 samples
- Priority Road - 1890 samples
- Keep Right - 1860 samples

Traffic signs with fewer samples:
- Speed limit (20km/h) - 180 samples
- Dangerous curve to the left - 180 samples
- Go straight or left - 180 samples
- Pedestrians - 210 samples
- End of all speed and passing limits - 210 samples

It can be seen that there is an uneven number of samples for each traffic sign. Between the sign with most samples and the one with less samples, there are **1830** samples! This is something to consider in the design of the classification pipeline since this class imbalance could bring wrong classification results because the model would be reflecting the underlying class distribution.

Here are some samples from the data set.
![dataset]

Now, let's deep dive into our pipeline for traffic sign classification!

---

## Design and Test a Model Architecture

### Pre-process the Data Set

As a first step, I decided to convert the images to grayscale because it reduces model complexity and also because at the end, patterns, brightness, contrast, shape, contours shadows, and other image properties, are well captured by gray images without extra costs. Of course, use of coloured images will depend mainly on the task we want to solve,whether we need the extra information provided by the RGB channles or not will be part of the approximation.



### Model Architecture

Training with coloured images and using the original LeNet architecture introduced by LeCun et al. in their 1998. This yield to an accuracy of 88%. For this reason I decided to start with the basics, data set pre-processing.
Model hyperparameters:

EPOCHS = 10
BATCH_SIZE = 128
n_channels = 3 # For coloured images



### Model Training

### Solution Approach

## Test a Model on New Images

Here's a [link to my video result](output_images/project_video_out.mp4). 
The folder `output_images` contains examples of the output from each stage of the advanced lane detection pipeline, including videos.

## (Optional) Visualizing the Neural Network

---

## Discussion

There are som drawbacks! 
This project has been challenging and of course very interesting. Some exploration and update can be performed in the thresholdins process, thinking on combining different filters. For this project I selected the HLS to work with, but a combination might yield in better results during lane estimation.

Also, the warping stage could be dynamic instead of fixed tuned, we have to somehow ensure straight lines! Moreover, under some scenarios (on challenging videos) the current version lane detection is not 100% accurate, there are bigger shadows, cars, irregular street conditions, etc. to be highly considered. These conditions can be tackled by storing information related to previous video frames, thus in cases where lanes jump drastically or get lost, they can be validated by using past references, mean values, etc., making the detection cleaner.

Thank you for reading this report.

_Daniel_


[class_freq]: report_images/Class_Freq.png
[dataset_dist]: report_images/Traffic_Signs_Distribution.png
[dataset]: report_images/Explore_DS.png