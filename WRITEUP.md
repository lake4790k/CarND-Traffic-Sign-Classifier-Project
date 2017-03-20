# **Traffic Sign Recognition** 

## Writeup


[//]: # (Image References)

[image1]: ./doc/seq.png "Sequence of images"
[image2]: ./doc/randoms.png "Random images"
[image3]: ./doc/unbalanced.png "Unbalanced classes"
[image4]: ./doc/preproc.png "Preprocessing"
[image5]: ./doc/random_preproc.png "Random preprocessed images"
[image6]: ./doc/new5.png "Unseen signs"
[image7]: ./doc/new_provs.png "Unseen probs"


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Cells 3-7.

![alt text][image1]

The images seem to be consequtive frames of videos from a moving vehicule, so there are multiple images of the same sign getting scaled up as it gets closer. Hence there will be no need to enrich the test set with scaled images. Rotations could be intersting though.

![alt text][image2]

The images have quite different contrast/brightness levels, probably we will need preprocessing to equalize the brightness/contrast accross all images.

![alt text][image3]

The test set is not balanced between the classes. I need to make sure the test set will be balanced as net would be biased to just pick the class with the most training examples.

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Cell 8.

My preprocessing
* balances the number of training examples for each class (training set only)
* per image converts to greyscale (training, test, valid set) It's counterintuitive that loosing the color information improves, but the linked paper suggests so
* per image uses skimage.equalize_hist to enhance the dark images (training, test, valid set)
* per image applies a normally distributed rotation around 0 degrees to each image (training set only), so oversampled images are all different

![alt text][image4]

I checked the training, validation and test images that they have a uniform preprocessing, eg:

![alt text][image5]


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the supplied validation and test sets. I did not do cross validation as training the bigger nets was time consuming even on the AWS GPU instances. Also the requirement was to achieve a given accuracy on the validation set supplied, so it made no sense to modify the validation set.

The training set was not balanced between the classes. To fix this during the preprocessing I made sure all classes have exactly 2000 images by oversampling the smaller classes. I also applied random rotations on the images, so it's not exactly the same image that is repeated many times.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Cells 14-16.

I tried two architectures. The first was an implementation of the LeCunn paper that was mentioned in the notebook with the skip connection from the 1st convolutional layer to the linear classifier based on LeNet. I also used dropout. Unfortunately I couldn't get the kind of accuracies mentioned in the paper with this architecture (only mid 97%).

The second approach was a deeper model with stacking convolution layers without pooling and with dropout.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 greyscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, 16 features 	|
| RELU					|												|
| DropOut				|												|
| Convolution 5x5     	| 1x1 stride, valid padding, 16 features 	|
| RELU					|												|
| DropOut				|												|
| Convolution 5x5     	| 1x1 stride, valid padding, 32 features 	|
| RELU					|												|
| DropOut				|												|
| Convolution 5x5     	| 1x1 stride, valid padding, 32 features 	|
| RELU					|												|
| DropOut				|												|
| Convolution 5x5     	| 1x1 stride, valid padding, 64 features 	|
| RELU					|												|
| DropOut				|												|
| Convolution 5x5     	| 1x1 stride, valid padding, 64 features 	|
| RELU					|												|
| DropOut				|												|
| Convolution 5x5     	| 1x1 stride, valid padding, 128 features 	|
| RELU					|												|
| DropOut				|												|
| Fully connected		| 1024 neurons	    							|
| Fully connected		| 256 neurons	        						|
| Softmax				| 43 classes	        						|
|						|												|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Cells 17-18.

I used an initial learning rate of 0.001 with exponential decay.

I used a batch size of 128 which seemed to ensure an efficient workload on the AWS GPU.

With the enriched training set 100 (or less) epochs seemed sufficient to reach the potential of the networks I experimented with.

I used the AdamOptimizer, I didn't try any other as there were quite a few other hyperparameters to experiment with already.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Cell 19.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 99.3%
* test set accuracy of 97.5%

LeNet was the starting point. Basic LeNet does not achieve the required minimum accuracy obviously.
Then I added the skip connections to implement the multistage classifier from the mentioned LeCunn paper.

With the multistage skip connection network I could get a passing accuracy (~97.5%), but not as high as the paper stated.

When I started increasing the network size I saw that the training set accuracy was easily 100%, so the network couldn't learn any more, but the validation accuracy was much worse, so I overfitted the training set. To counter this I added L2 regularization for the network weights in the cost and also added dropout to every layer. This really helped to prevent overfitting and the validation accuracy could go up much more.

Since I couldn't get a really high accuracy (>99%) with the skip connection two layer convolutional network, I tried to just go deeper with the convolutions without pooling layers and no skip connections.

A ran many experiments with both models on AWS GPUs to see what network size can be trained in a reasonable time frame. I used multiple AWS instances in parallel to see what models/hyperparameters worked well.
 
My final solution followed the general avice "deeper is better". Since not only the training, but the validation and test accuracies are high as well, I believe the network didn't just learn all the training examples, but achived a more generalized knowledge. Although the test set accuracy was only 97.5% compared with 99.3%, so could be still better.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I picked 5 traffic signs from Google Streetview:

![alt text][image6]

The first and last should be easy. The fourth one was rotated, but I trained on rotations as well, so should still not be a problem.

The second and especially the third pictures are taken from the side and I didn't add these transformations to the preprocessing, so might be a problem.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Cell 20.

My model got all the signs right, except for the most difficult 3rd (so 80% accuracy)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Cell 21.

![alt text][image7]

The network is quite certain about the easy 1st and 5th images. And even about the rotated 4th one as it saw plenty of generated rotations.

The 2nd and 3rd images are taken from the side. The dataset was surely recorded from a forward facing camera, so these are more problematic. Still the network is right and confident about the 2nd one.

The 3rd one was really difficult, the network never saw anything like that and actually all five of its guesses are wrong. This is kind of OK as long as we want to use a forward facing camera, but a warning as well that the network is not a magician, it only works with cases it was trained for and fails in untrained, unexpected circumstances that are still easy for us humanss.


