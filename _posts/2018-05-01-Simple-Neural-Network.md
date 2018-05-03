---
layout: post
title: "Simple Neural Networks- My Understanding"
date: 2018-05-01
---

Neural Network is one of the ways to make a machine learn from data and is a type of Machine Learning technique.There are multiple Machine learning techniques but the power of the model is limited by the choice of algorithm say a decision tree algorithm or a Bayesian algorithm. In Neural netowrks there is no specific associated learning algorithm i.e the rules of learning are not explicitly mentioned and hence it gives the power to identify non-linear patterns in data where the relationship between the input and output is not straightforward.
    A simple neural network is illustrated below along with snippets of Python code.
  
NEURAL NETWORK INITIALIZATION

![Network1](/images/NN-Step1.png){:class="img-responsive"}  

{% highlight python%}
#Initialising variables
i = np.array ([[0.05,0.1]]) # Single record input array
o = np.array ([[1]]) # Output for the Single training example
output_neurons = 1
hidden_neurons = 2
wh= np.array ([[0.15,0.25],[0.2,0.3]]) # 2 x 2  matrix  - number of input features x number of hidden nodes
wout=np.array([[0.40],[0.45]]) # 2  x 1 - number of hidden nodes x number of output nodes
{% endhighlight%}

FORWARD PASS
![Network1](/images/NN-Step2.png){:class="img-responsive"}  

{% highlight python%}
#Hidden Layer Forward Pass
def sigmoid(x) :   return 1/(1+np.exp(-x))
hidden_layer_input = np.dot(i,wh) # returns h1_input and h2_input
hidden_layer_output = sigmoid(hidden_layer_input) #returns h1_output and h2_output
{% endhighlight%}

![Network2](/images/NN-Step3.png){:class="img-responsive"}  

{% highlight python%}
#Output Layer Forward Pass
out_layer_input = np.dot(hidden_layer_output,wout)  # returns o1_input
out_layer_output = sigmoid(out_layer_input) #returns o1_output
{% endhighlight%}

After the First Forward Pass the Error from the system is as below

$$Error = 1/2 *\\((o - out_layer_output)^2\\)$$ = (1-0.606) = 0.394
