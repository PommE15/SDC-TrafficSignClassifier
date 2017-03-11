# **Traffic Sign Recognition** 

## **Submission Files**

* The [Traffic_Sign_Classifier.ipynb](https://github.com/PommE15/SDC-...) notebook
* A PDF export of the project notebook named [report.pdf](...)
* The [assets folder](https://github.com/PommE15/SDC-TrafficSignClassifier/tree/master/assets) (any additional files used for this project but not [GTSD](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) or [traffic-sign-data.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip))
* My [writeup report](https://github.com/PommE15/SDC-TrafficSignClassifier/blob/master/writeup.md) as a markdown here


## **Instruction of Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

The writeup below is based on the [rubric points](https://review.udacity.com/#!/rubrics/481/view).  

[//]: # (Image References)

[image2a]: ./assets/table_data_summary.png "Data Table"
[image2b]: ./assets/gallery_class_samples.png "Gallery"
[image2c]: ./assets/bars_class_distribution.png "Class Distribution"
[image3a]: ./assets/imgs_grayscale.png "Grayscale"


## **Writeup report**

### Dataset Summary & Exploration

#### 1. Dataset summary
In the code, I used python's numpy methods to calculate summary statistics of the traffic signs dataset. Please refer to code cell [2] in the IPython notebook.

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory Visualization
In [3] and [4] code cells of the IPython notebook, it displays: 
a) a data table of the summary of training and validation data mapping with class labels,  
b) a gallary of a random sample of 43 classes in training data, and 
c) a bar chart that shows the counting distrubution of these classes.

![alt text][image2a]
...
![alt text][image2b]
![alt text][image2c]

As we can see in the bar chart, some classes (in red color) have less than 500 samples. This may lead to issues such as overfit or increase false classification. So, let's come back to check these later.  


### Design and Test a Model Architecture

#### 3. Data Preprocessing
There are two stages of data preprocessing I consider to apply. 
* grayscaling and normalization
* data augmentation

In the first stage, I converted the images to grayscale. Although color could play an important role in different categories of traffic signs for human preception; qualities such as size, brightness, saturation of sample images could change RGB values easily and mislead our learning model. Then I normalized the images with min-max scaling to a range of [0.1, 0.9] for activation function. Both techniques also improve constrast casued by glare or darkness. Please refer to code cell [5]. Note that other feature scaling and normalization methods are available for different purposes at [cs231n.github.io](http://cs231n.github.io/neural-networks-2/#datapre).

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image3a]

The second stage is data augmentation. I haven't decided whether using it or not. I'll come back to this after a few training and validation experiments. This will be my reference of [how to generate additional data](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3#.ywt6bxs5d). 

Note that condsidering both computer power and small dataset, I decided not to use corss validation for this project.


#### 4. Weight Initialization

I used the tf.truncated_normal() function to initialize the weights and set all bias to 0. Using the default mean and standard deviation from tf.truncated_normal() is fine. However, tuning these hyperparameters can result in better performance ...

In practice, the current recommendation is to use ReLU units and use the w = np.random.randn(n) * sqrt(2.0/n)

#### 5. Model Architecture 
Here are two diagrams and tables describing my experiments models. 
The code for these models is located in [*] cell of the ipython notebook. 
These models are based on the [LeNet example](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb) we learned in the class and 
the [example paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that was mentioned in the notebook.

The LeNet model consisted of the following layers:

![alt text][image5a]

| Layer         		|     Description	        					  | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   			  | 
| Convolution 5x5 | 2x2 stride, valid padding, outputs 32x32x64 	|
| RELU				|												|
| Max pooling	    | 2x2 stride,  outputs 16x16x64  |
| Convolution 5x5	| 2x2 stride, valid padding, ... |
| RELU				|												|
| Fully connected	| etc. |
| RELU				|												|
| Dropout	|												|
| Fully connected	| etc. |
| RELU				|												|
| Dropout	|												|
| Softmax				     | etc. |

The second model consisted of the following layers:

![alt text][image5b]

...

#### 6. Training
The code for training the model is located in the [*] cell of the ipython notebook. 
To train the model, I experimented with the following paremeters:

- k outputs
- dropout
- multi-scale
- activation function (sigmoid, ..., relu)
- regularization (L2?, dropout)
- optimizer (gd, adagrad, adam)
- learning rate
- batch size
- number of epochs 

By doing this, I wasn't trying find out the best model or hyperparemters.
I would just like to see how these tunings influence the acccurancy. 

- spreadsheet links and
- some charts

#### 7. Solution Design
Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 8. Acquiring New Images
Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 9. Performance on New Images
Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 10. Model Certainty - Softmax Probabilities
Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
