![](https://habrastorage.org/webt/lo/lw/cs/lolwcsuyedsd3yx3kgfc18ve3_y.png)

Stanford's [course cs231n](http://cs231n.stanford.edu/) "Convolutional Neural Networks for Visual Recognition" is one of the best ways to dive into Deep Learning in general, in particular, into Computer Vision. If you plan to excel in another subfiled of DL (say, NLP or Reinforcement Learning), we still recommend that you start with cs231n, because it gives you fundamental understanding and hands-on skills. Beware, the course is very challenging! 

To motivate you to work as hard as Stanford's students, here are actual applications that you'll implement in A3 - style transfer and class visualization. 

![alt-text-1](https://habrastorage.org/webt/ik/ny/o4/iknyo4fnkbokzoavq6nlsuitc6y.png "title-1") ![alt-text-2](https://habrastorage.org/webt/8t/go/qa/8tgoqaoa1vwmiuagfkx0i4nkjmm.png "title-2")

For the first one (on the left), you take a base image and a style image and "apply the style" to the base image (reminds of Prisma and Artisto, right?). The second example (on the right) is a random image, gradually perturbed in a way that a neural network classifies it more and more confidently as a gorilla. DIY Depp Dream, isn't it? And it's all math under the hood, it's cool to figure out how it all works. You'll get to this understanding if you pass cs231n, it'll be hard but at the same time an exciting journey from efficient kNN implementation to these fascinating applications. If you think that these two application are too eye-cathy (yes they are), then take a look at a picture above - a Convolutional Neural Network classifying images. That's the basics of how machines can "see" the world. The course will teach you both how to build such an algorithm from scratch and how to use modern tools to run State-of-the-Art models for your tasks. 

# Assignment 1
 - [k-Nearest Neighbor (kNN) exercise](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment1/knn_solution_yorko.ipynb) + [comments on some derivations](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment1/knn_comments_yorko.ipynb)
 - [Softmax classifier](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment1/softmax_solution_yorko.ipynb)
 - [Multi-class SVM](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment1/svm_solution_yorko.ipynb)
 - [Two-layer net](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment1/two_layer_net_solution_yorko.ipynb)
 - [More features](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment1/features_solution_yorko.ipynb)

# Assignment 2
 - [Fully-Connected Neural Nets](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment2/FullyConnectedNets_solution_yorko.ipynb)
 - [Batch Normalization](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment2/BatchNormalization_solution_yorko.ipynb	)
 - [Dropout](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment2/Dropout_solution_yorko.ipynb)
 - [Convolutional Networks](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment2/ConvolutionalNetworks_solution_yorko.ipynb)
 - [PyTorch convnet](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment2/PyTorch_solution_yorko.ipynb)

# Assignment 3
- [Image captioning with RNNs](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2018/blob/master/assignment3/RNN_Captioning_solution_yorko.ipynb?flush_cache=true)
- [Image captioning with LSTMs](http://nbviewer.ipython.org/urls/raw.github.com/Yorko/stanford_cs231n_2018/master/assignment3/LSTM_Captioning_solution_yorko.ipynb)
- [Network Visualization (PyTorch)](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2019/blob/master/assignment3/NetworkVisualization-PyTorch_yorko.ipynb)
- [Generative Adversarial Networks (PyTorch)](https://nbviewer.jupyter.org/github/Yorko/stanford_cs231n_2019/blob/master/assignment3/Generative_Adversarial_Networks_PyTorch_yorko.ipynb)
- [Style Transfer (PyTorch)](http://nbviewer.ipython.org/urls/raw.github.com/Yorko/stanford_cs231n_2019/master/assignment3/StyleTransfer-PyTorch_yorko.ipynb)

## Passing cs231n together within the [OpenDataScience](http://ods.ai) community
Next start - from **02.12.2019** till **08.03.2020**

**Main links**
- The [course](http://cs231n.stanford.edu/) itself 
- Video-lectures, youtube [channel](https://goo.gl/pcj7c8). Prerequisites are given in the 1st lecture  
- [Syllabus](http://cs231n.stanford.edu/syllabus.html) with assignments
- Unofficial [lecture notes](https://github.com/mbadry1/CS231n-2017-Summary) by [Mahmoud Badry](https://github.com/mbadry1)
- For Russian-speaking audience, a good alternative is the Deep Learning course [dlcourse.ai](https://dlcourse.ai/) lead by Simon Kozlov, [sim0nsays](https://twitter.com/sim0nsays?lang=en)

**Assignments**

There are 3 big and tough assignments in this course. The biggest challenge is to actually do these assignments on your own because solutions are easily accessible anywhere on the Internet and are even shared by me in this repo. 

**Competitions & projects**

In the original course they've got [projects](http://cs231n.stanford.edu/project.html). You can also complete one, but actually, lectures and assignments is already a good workload. So my advice is to first cope with assignments and then you can go on with pet projects or Kaggle.

**GPUs**
For some parts of the 3rd assignment, you'll need GPUs. Kaggle Kernels or Google Colaboratory will do.

**Plan**

- 02.12.19 – 08.12.19. [Lecture 1](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) and [Lecture 2](https://www.youtube.com/watch?v=OoUX-nOEjG0&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=2)
- 09.12.19 – 15.12.19. [Lecture 3](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=3) and [Lecture 4](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=4)
- 16.12.19 – 22.12.19. [Lecture 5](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=5) and [Lecture 6](https://www.youtube.com/watch?v=wEoyxE0GP2M&index=6&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- 22.12.19 – **A1 due**. You can discuss it in the **#class_cs231n** channel in Slack
- 23.12.19 – 12.01.20. [Lecture 7](https://www.youtube.com/watch?v=_JB0AO7QxSA&index=7&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) and [Lecture 8](https://www.youtube.com/watch?v=6SlgtELqOWc&index=8&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- 13.01.20 – 26.01.20. [Lecture 9](https://www.youtube.com/watch?v=DAOcjicFr1Y&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=9) and [Lecture 10](https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=10)
- 27.01.20 – 09.02.20. [Lecture 11](https://www.youtube.com/watch?v=nDPWywWRIRo&index=11&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) and [Lecture 12](https://www.youtube.com/watch?v=6wcs6szJWMY&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=12)
- 09.02.20 – **A2 due**. You can discuss it in the **#class_cs231n** channel in Slack
- 10.02.20 – 23.02.20. [Lecture 13](https://www.youtube.com/watch?v=5WoItGTWV54&index=13&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) and [Lecture 14](https://www.youtube.com/watch?v=lvoHnicueoE&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=14)
- 24.02.20 – 08.03.20. [Lecture 15](https://www.youtube.com/watch?v=eZdOkDtYMoo) and [Lecture 16](https://www.youtube.com/watch?v=CIfsB_EYsVI&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=16)
- 08.03.20 – **A3 due**. You can discuss it in the **#class_cs231n** channel in Slack
