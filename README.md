Notes for ML

What is decision tree?
Yes no question, 
Continuous > cutoff question
MC / rank > category question

Root node > internal nodes > Leaf nodes

decision tree for every feature 
Measure impurity - Gini
￼
￼
Lowest Gini to split
￼
Split again and use lowest Gini
￼
If impurity is lower before splitting, make it a leaf node
￼
Continuous data
1. Rank the data
2. Calculate average between data
3. Calculate impurity for each average
￼
Rank data 
Calculate Gini for every rank

MC Data
Calculate Gini even for Combination
￼
Automatic feature selection
If feature never gave reduction in impurity,
Would not use the feature

Control overfit for tree
Create a threshold such that impurity reduction has to be large enough

Missing data, fill mode or use similar feature as a guide for categorical data
￼
Use regression for continuous data


What is random forest?
Tree only works great with their data
Forest more flexible
1. Boostrapped dataset
2. Random subset of features

Build a random forest
1. Randomly select sample, allow to pick same sample more than once
 2. Randomly select 2 features for the root node and choose one with lower impurity
3. Randomly select 2 candidates to build tree (random subset of variables)

Use a random forest
Choose more votes prediction
Bootstrapping data + aggregate to make decision = bagging
Out of bag dataset = not in training, only run on tree build without it
Out of bag error

Can compare random forest built with only 2 variables and with 3 variables and choose the most accurate one with how many features

What is Gradient Boosting?

Why xgboost?
Training fast, C++ backend, Parallel


What is AdaBoost?
1. 1node + 2leaves = Stump (Weak learners)
2. Some stumps have larger weight 
3. Order of tree （take mistake into account）

Decision tree for every features
First stump = lowest Gini index (impurity)
Sample weight = error  / total (add up to 1)
￼
Increasing the weight of incorrect sample
Decreasing the weight of correct sample

Amount of say = 0.5 log [(1-total error)/total error]
￼
￼
Amount of say is the same for all sample 

Increasing sample weight for incorrectly classified sample
￼
￼
More than old one

Decreasing sample weight for correctly classified sample
￼
￼
Less than old one
￼
Use modified Norm weight to make second stump

In theory,
Use sample weights to calculate weighted Gini indexes to determine next variable (weighted mini index would put more emphasis on correctly classifying the incorrect sample)

Alternatively create new dataset
￼
￼
Pick random number and add sample to the new collection until the new collection Is the same size
Give all samples equal weight
￼
Add up the amount of say for both stumps
￼



Hadoop and Spark
Hadoop, database, distributed, no need to buy expensive server hardware, index and track data, high processing efficiency. HDFS data storage, 有MapReduce (唔洗用spark), MapReduce分步處理 每次處理後的數據都寫入到磁盤上

Spark, processing data for distributed data, not for storage, faster than MpaReduce, Spark同時處理, 快spark 10x 這些數據對象既可以放在內存，也可以放在磁盤
