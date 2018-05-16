---
layout: post
title: "Intuition behind dot product of Input and Weight matrices in Image Classification"
date: 2018-05-16
---

 What happens when we do a dot product of the weight matrix (denoted as W)and the input matrix(denoted as X)?
 Let us assume that the network is completely trained and W is learnt for each of the classes(say Rose/Leaf/Sea)
 For the sake of simplicity we are dealing with an image of 1 pixel(R,G,B) i.e a 1x3 vector.
 The W for the classes is of the dimension(i,j) = 3,3.i denoting RGB values and j denoting each of the 3 categories). 
 
 
 
 We can imagine the process of multiplying the input vector with the weight matrix as below.
 Step 1: The input vector X(Category:Leaf) is multiplied with the column vector(Rose) of Weights.Imagine this as superimposing the Rose   template(RGB values for the Rose Column)from the learned weight matrix  on the Input vector.Each of the element in the input vector is multiplied with each element in the Weight vector(for Category Rose) giving 3 values which we can assume to be the strength of the RGB values (notionally) after sumperimposing one image over the other.The resulting 3 values 
 
