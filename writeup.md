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
[image2b]: ./assets/bars_class_distribution.png "Class Distribution"
[image2c]: ./assets/gallery_class_samples.png "Gallery"
[image2d]: ./assets/gallery_train.png "Gallery of Train Examples"
[image2e]: ./assets/gallery_valid.png "Gallery of Valid Examples"
[image3a]: ./assets/image_preprocess.png "Hue good"
[image3c]: ./assets/image_hue_tragedy3.png "Hue bad"
[image3b]: ./assets/image_grayscale.png "Grayscale"
[image5a]: ./assets/diagram_leNet.jpg "Diagram LeNet"
[image5b]: ./assets/diagram_multiScale.jpg "Diagram Multi-Scale"
[image8a]: ./assets/test_keepRight.png ""
[image8b]: ./assets/test_limit60.png ""
[image8c]: ./assets/test_priorityRoad.png ""
[image8d]: ./assets/test_stop.png ""
[image8e]: ./assets/test_yield.png ""


## **Writeup report**

### Dataset Summary & Exploration

#### 1. Dataset summary 
In `[2] code cell`,
I used python's numpy library to calculate summary statistics of the traffic signs data set.

* The size of training set is 34,799
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory Visualization
In `[3], [4], and [5] code cells`,
there are five visualizations of data set.

* Data table of the summary of training and validation data mapping with class labels
* Bar chart that shows the counting distrubution of these classes  

![data table][image2a]
...
![bar chart][image2b]

In the table and bar chart, I would like to check the size of both training and validation sets and how they are splitted. These two visualizations show that, for example, the split ratio is in a range between 10-20%, the smallest sets (label 0, 19, 37) of training/validation data have less than 200/30 samples, and less than half of classes have more than 1000 training samples. Is it relatively small for a CNN model? Perhaps, I need more samples via data augmentation??

Now let's have a look at the images.

* Gallary of a random sample of 43 classes in training data
![gallary 43 classes][image2c]

I picked label 19 to take a deeper look at samples in one class. 

* Gallary of the training set
![gallary train][image2d]
* Gallary of the validation set
![gallary valid][image2e]

From galleries above, I observed that these images were likely taken from 30fps video clips. In every 30 clips, the same traffic sign zooms in and out, the hue/saturation/brightness (HSB) differs, and some of images are neither straight nor originally square. The validation set, in particular, there is only 1 video clips in class label 19, which is potential to issues such as overfitting or false classification.

### Design and Test a Model Architecture

#### 3. Data Preprocessing
In `code cell [1]`, all images were loaded by pickle library which contains resize version of 32x32 while original ones vary from 15x15 to 250x250.

In `code cell [6]`, I applied grayscaling and normalization of min-max scaling as preprocessing steps. Here is a list of data preprocessing steps that I consider to apply: 
* grayscaling (or huescaling)
* normalization 
* data augmentation (or cross-validation)

First of all, I converted the images to grayscale. Although color could play an important role in different categories of traffic signs for human preception; qualities such as size and HSB of sample images could easily change RGB values and mislead our learning model. On the other hand, huescaling is another interesting color transformation step to try, it does enhance the recognition in some cases, however, it could also over emphasize or simplify shapes of a image based on a few tests I performed. Here are two examples.

![img with hue][image3a]
![img with hue][image3c]

As the second step, I normalized the images with min-max scaling to a range of [0.1, 0.9]. The range is chosen because it was also used in the Tensorflow Lab we worked on. I'm not sure if this was chosen in the Lab for a specific reason (activation function?) or something else. How about ranges such as [-1, 1] and [0, 1]? Note that other feature scaling and normalization methods are available for different purposes at [cs231n.github.io](http://cs231n.github.io/neural-networks-2/#datapre).

Here is an example of a traffic sign image before and after grayscaling and normalization. Based on my tests, both steps improve recognition of a image, espicially, constrast casued by glare or darkness.

![img of greyscale][image3b]

As for the data augmentation step, I haven't decided whether using it or not. I'll come back to this after a few training and validation experiments. In case I need it, [how to generate additional data](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3#.ywt6bxs5d) will be my reference. 

Last but not lease, condsidering both computer power and the intrinsic factor that some classes of samples are lack of variations in images, I decided not to use either cross-validation or mixing (training and validation sets) and respliting them for this project.


#### 4. Initialization
In `[7] code cell`, to initialize weights and bias, I used the tf.truncated_normal() function. I set all bias to 0 and initialized weights with mean=0 and stddev=0.1. Note that, in practice, the current recommendation is to give the initialization w = np.random.randn(n) * sqrt(2.0/n), this is used in the specific case of neural networks with ReLU neurons. [[See reference]](http://cs231n.github.io/neural-networks-2/#init)


#### 5. Model Architecture 
In `[8] and [9] code cells`, I experimented two architectures and their parameters.

| architecture | feature type | application |
|-------|-------|-------|
| [LeNet-5](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb) | single (SS) | zip codes and digits |
| [Multi-Scale features](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) |  single or multi-scale (MS) | traffic sign recognition |

They both has layers as:
* `INPUT -> [5x5 CONV -> RELU -> 2x2 POOL]*2 -> [FC -> RELU -> DROPOUT]*2 -> FC`
* `POOL` uses max pooling for subsampling 
* `RELU` is the activation function
* `DROPOUT` is the regularization

Here are two diagrams of these architectures including their layer sizes and an example set of parameters:
* LeNet-5
![leNet][image5a]
* Multi-Scale features
![multi-scale][image5b]

For the architecure parameters, I tried a few sets of
* k outputs, i.e:

|       |    k1 |    k2 |    k3 |    k4 |
|-------|-------|-------|-------|-------|
| set 1 |     6 |    16 |   120 |    84 |
| set 2 |    38 |    64 |   100 |    50 |
| set 3 |    38 |    64 |   100 |   100 |
| set 4 |   108 |   108 |   100 |   100 |

in `CONV` (k1, k2 feature maps) and `FC` (k3, k4 neurons) layers with SS architecture for further experiemnts. 


#### 6. Training
In `code cells [10] and [11]`, to train the model, I experimented with the following (hyper)paremeters:

| parameters | values |
|-------|-------|
| batch size | 128, 256, 512 |
| number of epochs | up to 30  |
| optimizer | gradient descent, adagrad, and adam |
| learning rate | 0.001, 0.005, 0.01, 0.1, ... |
| keep prob | 0.4, 0.5, 0.6 or 0.7 in `DROPOUT` |

By doing this, I wasn't trying find out the best model or hyperparemters since some parameters seem to affect each other. My purpose was to observe how these tunings influence the acccurancy and use it as a base for the solution design. Here is the [spreadsheet](https://docs.google.com/spreadsheets/d/1ywtsyiECjC3c9LggXh1KTzy_Qqplh6WmpMLmK8KtIIU) that recorded part of the results. From experiment results, I observe that:

* **batch size** is better as 128,
* **epoch** is good between 15 to 25 depends on model complexity,
* either **adagrad and adam optimizer** gives good results but they are suitable for different ranges of learning rate, for example, adagrad around 0.1 and adam around 0.001,
* **keep prob** is lower in more sophisticated layer model than those with simplier layers; however, 0.5 is always a good start.


#### 7. Solution Design
In `code cells [12]`, I test one of models I experiemnted earlier.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

My iterative approach:
* What was the first architecture that was tried and why was it chosen?
> I chose LeNet-5 as the first architecture since this was taught in the clasee.
* What were some problems with the initial architecture?
> ...
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

![alt text][image8a] ![alt text][image8b] ![alt text][image8c] 
![alt text][image8d] ![alt text][image8e]

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
