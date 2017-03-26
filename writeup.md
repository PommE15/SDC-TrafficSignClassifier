# **Traffic Sign Recognition** 

## **Submission Files**

* The [Traffic_Sign_Classifier.ipynb](./Traffic_Sign_Classifier.ipynb) notebook
* A PDF export of the project notebook named [report.pdf](./report.pdf)
* The [assets folder](./assets) (any additional files used for this project but not [GTSD](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) or [traffic-sign-data.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip))
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
[image8a]: ./assets/lines_results_finals.png ""
[image8a]: ./assets/test_limit60.png ""
[image8b]: ./assets/test_priorityRoad.png ""
[image8c]: ./assets/test_yield.png ""
[image8d]: ./assets/test_stop.png ""
[image8e]: ./assets/test_keepRight.png ""
[image8f]: ./assets/test_all.png ""
[image10]: ./assets/gallery_featuremaps.png ""

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

| outputs |   k1 |   k2 |   k3 |   k4 |
|---------|------|------|------|------|
|   set 1 |    6 |   16 |  120 |   84 |
|   set 2 |   38 |   64 |  100 |   50 |
|   set 3 |   38 |   64 |  100 |  100 |
|   set 4 |  108 |  108 |  100 |  100 |

in `CONV` (k1, k2 feature maps) and `FC` (k3, k4 neurons) layers with SS architecture for further experiemnts. 


#### 6. Training
In `code cells [10] and [11]`, to train the model, I experimented with the following (hyper)paremeters:

| parameters | values |
|-------|-------|
| batch size | 128, 256, 512 |
| number of epochs | up to 30  |
| optimizer | Gradient Descent, Adagrad, and Adam |
| learning rate | 0.001, 0.005, 0.01, 0.1, ... |
| keep prob | 0.4, 0.5, 0.6 or 0.7 in `DROPOUT` |

By doing this, I wasn't trying to find out the best model or hyperparemters since some parameters seem to affect each other. My purpose was to observe how these tunings influence the accuracy and use it as reference for the solution design. Here is the [spreadsheet](https://docs.google.com/spreadsheets/d/1ywtsyiECjC3c9LggXh1KTzy_Qqplh6WmpMLmK8KtIIU/edit#gid=72528419) that recorded part of the iterative experiment. From the results, I observe that:

* **batch size** is better using 128
* **epoch** is good between 15 to 25 depends on model complexity
* either **adagrad and adam optimizer** gives good results but they are suitable for different ranges of learning rate, for example, Adagrad around 0.1 and Adam around 0.001 based on my experiment
* **keep prob** 0.5 is always a good start


#### 7. Solution Design
In `code cells [12]`, I tested on one of models I experiemnted earlier.

My final model (using model A, see below) results were:
* training set accuracy of 0.993
* validation set accuracy of 0.964
* test set accuracy of 0.941

My iterative approach:
* What was the first architecture that was tried and why was it chosen?
> I chose LeNet-5 as the first architecture just to try and make sure I have a running model. 
* What were some problems with the initial architecture?
> The layers of LeNet-5 architecture work well. However, this model was used for alphabets and digit recognition, the convolution layers might be too simple for image cognition. Thus, I referenced parameters of k outputs (feature maps and neurons) in the paper of Traffic Sign Recognition with Multi-Scale Convolutional Networks and this improved the accuracy.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
> Based on the reference paper, I also tried to add multi-scale features. The accuracy over epochs increase nicely (not too fast). However, this approach requires too much time for my compute power, I just tried a few setups in the paper and observe the accuracy curve over epochs. I didn't use it for my final model. On the other hand, I added dropout in `fc` layers in my models to avoid over fitting. The models performed better with dropout than without. I observed the accuracy difference between training and validation sets to tune the keep probability. 
* Which parameters were tuned? How were they adjusted and why?
> All parameters mentioned in section 6. training are tuned to observe the influence on results and for making decision for the final model. In particular, I chose the optimizer and batch size based on my experiment results. I also checked where and how often the accuracy drops over epochs to micro tune the learning rate.   
* What are some of the important design choices and why were they chosen? 
> Most of the choices were made by observing experiment results and other points mentioned earlier. 

Here are three models that I considered to use for the final design. All of them use single feature, Adam optimizer, and batch size 128. 

| parameters | model A | model B | model C |
|-------|-------|-------|-------|
| k outputs     | **38, 64, 100, 50** | 108, 108, 100, 100 | 6, 16, 120, 84 |
| keep prob     | **0.5** | 0.48 | 0.6 |
| learning rate | **0.001** | 0.0008 | 0.002 |
| epochs        | **16 (early stop)** | 18 (early stop) | 25 |
| session       | **./convnet** | ./convnet 3 | ./convnet 4 |

Note1. The epoch sets up to 25, and it triggers early stopping in conditions of both training accuracy > 0.99 and validataion accuracy > 0.96. Those two numbers are also used to avoid over fitting and they came from the iterative experiment.

Note2. A session file is saved for testing.

The results of these models can be found in this [spreadsheet](https://docs.google.com/spreadsheets/d/1ywtsyiECjC3c9LggXh1KTzy_Qqplh6WmpMLmK8KtIIU/edit#gid=685744898). Here is a line chart shows the accuracy over epochs:

![alt text][image7a]

Base on this final experiment, I chose **model A** as final model to test on new images.


### Test a Model on New Images
The implementation of this section is located in `[13] - [16] code cells`.

#### 8. Acquiring New Images

Here are five German traffic signs that I found on the web and google map:

![alt text][image8a] ![alt text][image8b] ![alt text][image8c] 
![alt text][image8d] ![alt text][image8e]

Here are how they look like after loading with pickle:
![alt text][image8f]

The second and fifth images might be difficult to classify because of the quality (resolution). The background of the second image could be confusing and the angle of the fifth image is not so straight.

#### 9. Performance on New Images with Softmax Probabilities (Prob)

In `code cell [14]`, the model was able to correctly guess all traffic signs. This could thanks to the number of training samples - 4 of the 5 traffic signs have more than 1000 samples, or the testing images are simply just too easy to be classified. In `code cell [15]`, all images are more than 97% sure about the traffic sign they are recognized.

Here are the results of the prediction with top 3 probabilities:

| Image	/	Prediction 1 | Prob 1 | Prediction 2 | Prob 2	| Prediction 3 | Prob 3	| 
|----------------------|--------|--------------|---------|--------------|---------| 
| **Speed limit (60km/h)** | .9998624 | Go straight or right | .0000833 | End of all speed and ... | .0000494 |
| **Priority road**     	 | .9704881 | Roundabout ... | .0290123 | End of no passing  | .0002145 |
| **Yield**					       | .9999999 | Ahead only  | .0000001 | Keep right  | 0 |
| **Stop**	      		     | .9915633 | Turn right ahead | .0048740 | No entry | .0017053 |
| **Keep right**			     | .9997202 | Turn left ahead | .0002099 | Yield | .0000678 |


### Visualize the Neural Network's State with Test Images
#### 10. Feature Maps Visualization

In the last `code cells [17]`, I plotted the output of the network's weight layers on top of a testing image. Here is an example of my network's c1 output feature maps on image of Speed limit (60km/h):

![alt text][image10]
