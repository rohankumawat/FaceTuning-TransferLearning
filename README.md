## What is Transfer Learning?
Transfer Learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. Or as wikipedia definition says, It is a research problem in Machine Learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.

It's just like starting a new career building on our past experiences. This deep learning technique enables developers to harness a neural network used for one task and apply it to another domain.

What we acquire as knowledge while learning about once task, we utilize in the same way to solve related tasks. The more related the tasks, the easier it is for us to transfer, or cross-utilize our knowledge. A simple example would be if we know how to ride a motorbike, we can learn how to ride a car also.
 
Let's just take an example: You want to identify countries food, but there aren't any publicly available algorithms that can do an adequate job. With transfer learning, you begin with an existing convolutional neural network commonly used for image recoginition of which food is this, and then you tweak it to train with the country name also. 

Transfer learning is the idea of overcoming the isolated learning paradigm and utilizing knowledge acquired for one task to solve related ones.

![Image of traditional machine learning and transfer learning](https://miro.medium.com/max/1400/1*9GTEzcO8KxxrfutmtsPs3Q.png)

### Benefits of Transfer Learning
* The benefits of transfer learning are that it can speed up the time it takes to develop and train a model by reusing these pieces or modules of already models.
* This helps speed up the model training process and accelerate results.
* When you've insufficient data for a new domain you want handled by a neural network and there is a big pre-existing data pool that can be transferred to your problem. So you might have only 1,000 images of food, but by tapping into an existing CNN such as VGG16, VGG19, etc trained with more than 1 million images, you can gain a lot of low level and mid level feature definitions.
* Not much computational power is required. As we are using pre-trained weights and only have to learn the weights of the last few layers.

![Image of Prediction accuracy or performance vs Training](https://lh5.googleusercontent.com/gFkTSCMrYNYgBuIF2u9EweYo-9kAdGabAi7Yx9oFYH69V0Nf28weEA8wzPDR3RIRZ1e1BLcwodj3Y3ZTTNPfkhgREmGtru6UvJLoxO6pvGjTa7G2tlUNmlRiUpGbTI-HMpe45pWl)

### Build a Face Recognition Model using transfer learning in Keras
We can use any model here to complete the task (VGG16 or VGG19 or MobileNet, etc). 
Here in FaceRecog_VGG16_224_224 and FaceRecog_VGG16_64_64, I've used VGG16.

#### Requirements:
* Keras (with Tensorflow backend)
* Computer Vision 
* Numpy

#### Data Requirement:
* The data must be stored in a particular format in order to be fed into the network to train. I've used ImageDataGenerator, available in keras to train our model on the available data.
* There must be a main data folder, inside that data folder, there must be a folder for each class of data containing the corresponding the images. The names of the folders must be the names of their respective classes.

#### Steps:
1. Importing the pre-trained model and removing the 3 fully connected layers at the top of the network.
2. Adding the dense layers.
3. Loading the data into ImageDataGenerators.
4. Training and evaluating model.

#### Files in the repository
* __Concept.ipynb file:__ In this file I've tried my best to make the viewer understand that what is happening and why it is happening in the other two files.
* __FaceRecog_VG16_224_224.ipynb file:__ In this file I've created the face recognition model by having the default input size of 224 x 224 pixels and then have tested the images also. Currently the model is around 69% accurate.
* __FaceRecog_VG16_64_64.ipynb file:__ In this file I've created the face recognition model by having the default input size of 64 x 64 pixels and then have tested the images also. Currently the model is around 81% accurate. By going for 64 x 64 pixels, the training of the dataset is much faster than 224 x 224 pixel model. 

![Architecture of VGG16 and Transfer Learning](https://i0.wp.com/appliedmachinelearning.blog/wp-content/uploads/2019/08/vgg16.png?resize=714%2C204&ssl=1)
