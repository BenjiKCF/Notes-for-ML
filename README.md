Notes for ML

What is decision tree?
Yes no question, 
Continuous > cutoff question
MC / rank > category question

Root node > internal nodes > Leaf nodes

decision tree for every feature 
Measure impurity - Gini


Lowest Gini to split

Split again and use lowest Gini

If impurity is lower before splitting, make it a leaf node

Continuous data
	0.	Rank the data
	0.	Calculate average between data
	0.	Calculate impurity for each average

Rank data 
Calculate Gini for every rank

MC Data
Calculate Gini even for Combination

Automatic feature selection
If feature never gave reduction in impurity,
Would not use the feature

Control overfit for tree
Create a threshold such that impurity reduction has to be large enough

Missing data, fill mode or use similar feature as a guide for categorical data

Use regression for continuous data


What is random forest?
Tree only works great with their data
Forest more flexible
	0.	Boostrapped dataset
	0.	Random subset of features

Build a random forest
	0.	Randomly select sample, allow to pick same sample more than once
 2. Randomly select 2 features for the root node and choose one with lower impurity
	0.	Randomly select 2 candidates to build tree (random subset of variables)

Use a random forest
Part1 
Choose more votes prediction
Bootstrapping data + aggregate to make decision = bagging
Out of bag dataset = not in training, only run on tree build without it
Out of bag error
Can compare random forest built with only 2 variables and with 3 variables and choose the most accurate one with how many features

Part2 
Missing data & sample clustering in forest

Missing data
	0.	Missing data in the original dataset
	0.	Missing data in a new sample that we want to categorise

	0.	Missing data in the original dataset
	0.	Fill in with Most common, Median value 
	0.	Fill in Proximity matrix 
 Keep track a similar samples 
If end up in the same leaf node, put a 1

	0.	After running second tree, add up the values

	0.	Sum up all the values for all trees

	0.	Divide each proximity value by no of trees

	0.	Using weighted frequency to calculate missing values (categorical)

Proximity weights 0.1 0.1 0.8
1/3 Yes, 2/3 No
Weight for yes = 1/3 * (0.1 / 0.1+0.1+0.8) = 0.03
Weight for no = 2/3 * (0.9 / 0.1+0.1+0.8) = 0.6 

	0.	Using weighted frequency to calculate missing values (continuous)

0.1*125+0.1*180+0.8*210 = 198.5

	0.	Then build random forest again 6 - 7 times.

Promixty = 1, distance = 0
Green one = distance matrix, can draw heatmap / MDS plot

Can draw heat map / MDS plot to show relationship

Missing data in a new sample we want to categorise
	0.	Create 2 copies, one label yes, one label no.
	0.	Then use iterative method
	0.	See which one is correct in more trees


What is AdaBoost?
	0.	1node + 2leaves = Stump (Weak learners)
	0.	Some stumps have larger weight (better, more weights)
	0.	Order of tree （take mistake into account）

Decision tree for every features
First stump = lowest Gini index (impurity)
Sample weight = error  / total (add up to 1)

Increasing the weight of incorrect sample
Decreasing the weight of correct sample

Amount of say = 0.5 log [(1-total error)/total error]


Amount of say is the same for all sample 

Increasing sample weight for incorrectly classified sample


More than old one

Decreasing sample weight for correctly classified sample


Less than old one

Use modified Norm weight to make second stump

In theory,
Use sample weights to calculate weighted Gini indexes to determine next variable (weighted mini index would put more emphasis on correctly classifying the incorrect sample)

Alternatively create new dataset


Pick random number and add sample to the new collection until the new collection Is the same size
Give all samples equal weight

Add up the amount of say for both stumps


What is Gradient Boosting?

Part 1 Algorithm
GB only makes a single leaf, (not stump)
represents initial guess for the all the samples. Like average value.

Like adaBoost, 
	⁃	this tree is based on previous mistake, but larger than a stump. (Max no. of leaves = 8-32 )
	⁃	Builds fixed size trees
	⁃	Scales the trees
	⁃	Until it has made the number of tress you asked for / no improvement
Unlike adaBoost,
	⁃	Each tree can be larger than a stump
	⁃	Scales the tree with the same amount

How to do a regression with gradient boosting 
	0.	Take average for target

	0.	Calculate Errors 
(observed - predicted) to get a Pseudo residual 

	0.	Build a tree to predict the residual (restricting the total number of leaves,)
Fever leaves than residuals

	0.	Replace residuals with average

	0.	Combine original leaves with the new tree.

Average target = 71.2
If Prediction = 71.2 + 16.8 = 88
Overfit to the data too well, Low bias, very high variance

Gradient boost deals with this problem by using a learning rate to scale the contribution.
	0.	Prediction = average weight + learning rate * first tree prediction
	= 71.2 + 0.1 * 16.8
Isn’t as good as before, but better than original leaf
Taking lots of small steps in the right direction result in better predictions. 
 
	0.	Build another tree with new pseudo residual

	0.	Combine it again to create a chain of tree until maximum specified or no improvement

Each time we add residual, residual is smaller.

Part 2 Maths

Input (x_i, y_i) i=1 to n and a differentiable Loss Function L(yi, F(x)), (usually MSE = 1/2 (observed - predicted)^2)

Makes the maths easier, derivative = negative residual

	0.	Find a predicted value minimise the sum of square residual, which is average and just a leaf




	0.	For m=1 to M: (M ＝ 

	0.	no of trees, usually 100)

2a. Calculate derivative of loss for every sample (This is why they call it gradient boosting) gives us the residual

-1*derivative = observed - predicted = residual


r_im = Pseudo residual, data_i, tree_m
Pseudo residual because it multiplies 0.5

r_1,1 = (88-73.3) = 14.7

2b. Fit a regression tree to the previous residual predict r_im  

R_jm = predicted residual, j = which leaf, m = tree

c. Calculate output value of each leaves
average the values in terminal node

Gamma_jm = value of gamma that minimise this summation (take previous prediction into account) (Only includes sample that is in the leaf)
Gamma = output value for each leaf

 j = which leaf, m = tree, j=1, m=1, 


Prediction = 73.3

Find gamma that minimise this equation
56 - 73.3 = -17.3
Gamma that minimise this equation is also -17.3
R1,1 has an output value of -17.3

2d. Update new prediction


Nu = v = learning rate

73.3 + 0.1 * 8.7 = 74.2 closer to 76, improvement

M is the no of trees(usually 100)
Step 3. Output F_M(x)

F_2(x) = F_0(x) + 0.1 * R1,1 + 0.1 * R1,2

Part 3 Classification

	0.	Initial Prediction
The initial prediction for individual is the log(odds) = logistic regression of average

Initial prediction = log(4/2) = 0.6931


Probability = 0.6667 = 0.7 > 0.5 Therefore classifying everyone = Yes

Residual = Observed - Prediction

Red and Blue dots are observed value, dotted lines are predicted values. 
Residual =  (1-0.7) = 0.3


	0.	Build a tree (In practice with 8-32 leaves)

	0.	Calculate the output values. (Log odds)   If single prediction in the leaf, needs transformation


 Previous probability is the same 

	0.	Combine the value

log(odds) prediction = 0.7 + (0.8 * 1.4) = 1.8
	0.	Then calculate Probability 
Probability = sigmoid(1.8) = 0.9

	0.	 Calculate new residual
	0.	Build new tree and calculate new output value


Output value = 2


Why xgboost?
Training fast, C++ backend, Parallel



Hadoop and Spark
Hadoop, database, distributed, no need to buy expensive server hardware, index and track data, high processing efficiency. HDFS data storage, 有MapReduce (唔洗用spark), MapReduce分步處理 每次處理後的數據都寫入到磁盤上

Spark, processing data for distributed data, not for storage, faster than MpaReduce, Spark同時處理, 快spark 10x 這些數據對象既可以放在內存，也可以放在磁盤

Notes for DL

XLNet
Better than BERT, GPT, ELMo
Autoregressive LM

Use x1 to predict x2, then use (x1, x2) to predict x3, etc.
Find theta that maximise the log likelihood
	⁃	GPT used mask to be AR
	⁃	ELMo use bidirectional LSTM, forward and backward to makes 2 hidden states. Independent training.

Advantage: Text generation / machine translation, from left to right, Autogressive is good at it. In Bert DAE, training and application are different, bad at generative.
Disadvantage: Cannot use 上文 下文 simultaneously.

Autoencoder LM
Denoising auto encoder (BERT)

Masked Language Model (MLM) mask 15% token replaced by <Mask>,  to predict original word.  Use words before and after a sentence. 

Advantage and disadvantage are exactly opposite to Autoregressive.

Advantage: Bidirectional, can see 上文 下文
Disadvantage:  <Mask> will cause difference between pretraining and fine-tuning. In fine-tuning, no <Mask> is used. Input noise exists between pretraining and fine tuning.

XLNet 
Uses AR with birdirectional
Hybrid of Bert, GPT2.0, transformerXL
Like BERT bidirectional,
Like GPT  with more high quality pretraining, 16 + 19 + 78G 
Like TransformerXL to handle long corpus

XLNet has pretraining and fine-tuning
Modified pretraining, use AR instead of DAE, no <Mask>, use word Context_before and Context_after.

Permutation Language model

Bert目前的做法是，给定输入句子X，随机Mask掉15%的单词，然后要求利用剩下的85%的单词去预测任意一个被Mask掉的单词，被Mask掉的单词在这个过程中相互之间没有发挥作用。

XLNet：对于输入句子，随机选择其中任意一个单词Ti，只把这个单词改成Mask标记，假设Ti在句子中是第i个单词，那么此时随机选择X中的任意i个单词，只用这i个单词去预测被Mask掉的单词。当然，这个过程理论上也可以在Transformer内采用attention mask来实现。
Randomly mask word. 愈前既字 mask掉比例愈高 
或者换个角度思考，假设仍然利用Bert目前的Mask机制，但是把Mask掉15%这个条件极端化，改成，每次一个句子只Mask掉一个单词，利用剩下的单词来预测被Mask掉的单词。，区别主要在于每次预测被Mask掉的单词的时候，利用的上下文更多一些

Xlnet hide the process of masking in the transformer 
	0.	If we are going to predict <Ti> which is x3.
	0.	Let Context_before (x1, x2) read x4
	0.	Random shuffle (x1, x2, x3, x4) and use the permutation for training
However it can’t shuffle sequence In Fine-tuning, can only be done in pre-training, so we have:

If we are going to predict <Ti> i_th word.
1. Input sequence is the same
	0.	In transformer, use attention to choose i-1 random words, and put in front of <Ti>, mask other words
Uses bidirectional self-attention model

3>2>4>1
2 can read 3
4 can read 3, 2
1 can read 3, 2, 4

CNN 
Lenet 
Convolution > maxpooling > fully connected

AlexNet 2012
Activation function = Relu (以前用tanh)
Data augmentation and dropout
Multi gpu

VGGNet 2014 (2nd place)
Simplified AlexNet, same no of  parameters,  more conv layer, longer training

InceptionNet GoogLeNet 2014 (1st place)
Less parameters than AlexNet

a) 1x1 cone
b) 1x1 conv > 3x3 conv
c) 1x1 cone > 5x5 conv
d) 3x3 max pooling > 1x1 conv

Wider network, more information from features
1x1 conv can reduce parameters
a) If 100x100x56 > filter 5x5x125  (stride 1, pad 2)> output = 100x100x125 
(conv param = 56*5*5*125 = 175000)

b) If 100x100x56 > filter 1x1x28 > filter 5x5x125  (stride 1, pad 2)> output = 100x100x125 
(conv param = 56*1*1*28 + *5*5*125 = 98068)

	0.	Change conv and maxpooling to inception block
	0.	Average pooling to replace fully connected layer
	0.	2 auxiliary classifier to avoid gradient vanishing

ResNet (2016)
Deeper != better because of gradient vanishing

用一句更簡短的話來說明，那就是 Residual Block 解決了當逐漸加深網路時，gradient 不能回流下一層的問題。透過 skip connection，而使上層作 back-propagation 時，如果此層的參數逼近於零時，仍可以選擇捷徑的路徑，跳過此層，直接回流至來源層。
Ensure back propagation 


H(x) = F(x) + Wx, W = conv to align the dimension of channel 
