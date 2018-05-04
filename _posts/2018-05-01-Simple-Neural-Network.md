---
layout: post
title: "Basic Neural Network"
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
![Network2](/images/NN-Step2.png){:class="img-responsive"}  

{% highlight python%}
#Hidden Layer Forward Pass
def sigmoid(x) :   return 1/(1+np.exp(-x))
hidden_layer_input = np.dot(i,wh) # returns h1_input and h2_input
hidden_layer_output = sigmoid(hidden_layer_input) #returns h1_output and h2_output
{% endhighlight%}

![Network3](/images/NN-Step3.png){:class="img-responsive"}  

{% highlight python%}
#Output Layer Forward Pass
out_layer_input = np.dot(hidden_layer_output,wout)  # returns o1_input
out_layer_output = sigmoid(out_layer_input) #returns o1_output
{% endhighlight%}

Error (E) is defined as

$$Error = 1/2 *(o - o1_{output})^2$$

BACK PROPAGATION

The objective of back propagation is to  find out the proportion to which each of the weights initialised have influenced the Error and adjust the weights so that the Error is minimised.The influence of each of the weights in arrays Wh and Wout can be found by taking the  partial derivative of Error w.r.t that weight element.

The influence of Wout(1,1) on the Error is depicted below

![Network4](/images/NN-Step5.png){:class="img-responsive"} 

As per the chain rule we have  the below formula.

$$\partial E/\partial W_{out(1,1)}$$ $$= \partial E / \partial o1_{output} * \partial o1_{output} / \partial o1_{input} * \partial o1_{input} /\partial W_{out(1,1)} $$ $$ = (o1_{output}-o) * o1_{output}*(1-o1_{output})*h1_{output}$$ $$ = (0.606-1) * 0.606 *(1-0.606) *0.507 $$ $$ = -0.0476$$ 


Similarly the influence of Wh(1,1) on the Error is as follows

$$\partial E/\partial W_{h(1,1)}$$ $$= \partial E / \partial o1_{output} * \partial o1_{output} / \partial o1_{input} * \partial o1_{input} /\partial h1_{output} * \partial h1_{output}/\partial h1_{input} * \partial h1_{input}/\partial w_{h(1,1)} $$ $$=(o1_{output}-o) *o1_{output}*(1-o1_{output})*W_{out(1,1)}*h1_{output}*(1-h1_{output})*i1$$ $$ = (0.606-1) * 0.606 *(1-0.606) * 0.4* 0.507*(1-0.507)*0.05 $$ $$ = -0.00047$$

The influence of each of the weights pertaining to a layer can be succintly written as
Error from that layer * slope of that layer * input associated with that weight element

{% highlight python%}
#Back Propagation
pE= out_layer_output-o # Differentiated w.r.t o1_output
delta_output = (pE * der_sigmoid(out_layer_output)
pd_output = delta_output* hidden_layer_output.T
delta_hidden = delta_output*wout*der_sigmoid(hidden_layer_output)
pd_hidden = delta_hidden* i.T
{% endhighlight%}

Learning rate is the rate at which we want the weights to be updated.For our working example 0.5(lr) has been chosen as the learning rate.The weights should be updated after calculating the error and gradient at every layer since the weight from the next layer have an impact on the values passed to the previous layer.

$$W_{out(1,1)} = W_{out(1,1)}- lr * \partial E/\partial W_{out(1,1)}$$
$$=0.4 - (0.5*-0.0476) $$ $$=0.4238 $$

Similarly
$$W_{h(1,1)} = W_{h(1,1)}- lr * \partial E/\partial W_{h(1,1)}$$
$$=0.15 - (0.5*-0.00047) $$ $$=0.1502 $$

{% highlight python%}
#Weight update
wout = wout - (lr*pd_output)
wh = wh - (lr*pd_hidden)
{% endhighlight%}

The above steps complete an epoch (a complete pass of forward and back propagation)and the weights wh and wout are updated.Multiple iterations will train the weight towards the expected output and will minimise the error.The complete code on network training based on pre-defined epoch is provided in the below link

[Jupyter notebook Link](https://github.com/MashNagesh/NeuralNetwork/blob/master/NN_trial_Single_class.ipynb)

The above blog post is inspired by

[A Step by Step Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

[Understanding and coding Neural Networks From Scratch in Python and R](https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/)

