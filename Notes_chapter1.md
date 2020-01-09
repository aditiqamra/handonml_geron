# Notes part 1

 - scikit-learn is a set of python libraries for machine learning
- TensorFlow is for distributed numerical computation. Train and run large neural networks efficiently by distributing the computations across potentially hundreds of multi-GPU (graphics processing unit) servers. 
- Keras is deep learning API that runs on top of Tensorflow, theano etc
right. Fast and easy to extend.
* ML can be defined as science/art of programming computers so that they can learn from data provided
* examples that system uses to learn from - Training set
* each training example is a training instance/sample
* ML can be used for 
    * complex problems with no good solution
    * changing data
    * multiple rules/fine tuning 
    * derive patterns
* Examples would be
    * Image classification - using CNNs 
    * Detecting patterns - eg tumor in scans - using CNNs
    * classification - NLP, text classification, RNN, CNN, transformers
        * Automatically classifying news articles
        * Automatically flagging offensive comments on discussion forums
    * text summarization - NLP. Summarise long documents automatically
    * create chatbot - NLP
    * forecasting problems e.g. company revenues - Regression (Linear, polynomial, SVM, Random forest)
    * If the above problem will use  sequences of past performance metrics, you may want to use RNNs, CNNs, or Transformers
    * Speech recognition - RNN, CNN, transformer
    * Anomaly detection
    * Segmentation - clustering
    * Dimensionality Reduction - merge several correlated features into one and identify newer few no of features ie. [feature extraction]
    * Recommender system - Neural net
    * Build an intelligent bot - Reinforcement learning 

### Types of Machine Learning Systems

* Whether or not they are trained with human supervision (supervised, unsupervised, semisupervised, and Reinforcement Learning)

* Whether or not they can learn incrementally on the fly (online versus batch learning)

* One more way to categorize Machine Learning systems is by how they generalize. Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do (instance-based versus model-based learning)


#### Supervised learning

* Training set has desired solution called [labels]
* e.g. 
    * classification
    * Regression - predict a numeric value e.g. car price given set of features e.g. mileage, age etc called [predictors] - R
* [attribute] and [features] are often used interchangeably. [attribute] means data type e.g. mileage while feature can have several context dependant meanings but generally means attribute plus its value
* Some regression algorithms can be used for classification as well, and vice versa. For example, Logistic Regression is commonly used for classification, as it can output a value that corresponds to the probability of belonging to a given class (e.g., 20% chance of being spam).

Types of supervised ML algos
- k-Nearest Neighbors
- Linear Regression
- Logistic Regression
- Support Vector Machines (SVMs)
- Decision Trees and Random Forests
- Neural networks

#### Unsupervised learning 

No labels are present. Examples -
* Clustering
    * K-Means
    * DBSCAN
    * Hierarchical Cluster Analysis (HCA)
* Anomaly detection and novelty detection
    * One-class SVM
    * Isolation Forest
* Visualization and dimensionality reduction
    * Principal Component Analysis (PCA)
    * Kernel PCA
    * Locally Linear Embedding (LLE)
    * t-Distributed Stochastic Neighbor Embedding (t-SNE)
* Association rule learning
    * Apriori
    * Eclat

#### Semi-supervised learning

partially labeled data - e.g. google photos face recognition, you only need to label one or a few photos for the rest of them to get unsupervised labeling

examples - deep belief networks built on top of restricted Boltzmann machines


#### Reinforcement learning

* learning system is called [agent] that observes and learns from data and get [penalties/rewards] in return
* then it must learn on its own - what is the best strategy called [policy]
e.g. AlphaGO game

#### Batch or offline learning
 cant learn incrementally
 must be trained using all available data at once
 not reactive
 needs more compute resources
 
 #### Online learning
 - can train system incrementally by feeding it data instances sequentally in mini batches
 - can learn on fly
 - limited compute resources
 e.g. stock market data 

[learning rate] : how fast the ML adapts to new data. if LR is high, it will quickly learn but it will also quickly forget old data. Conversly low LR, system will have more inertia i.e. learn more slowly but also less sensitive to noise in new data or outliers

A big challenge with online learning is that if bad data is fed to the system, the system’s performance will gradually decline. If it’s a live system, your clients will notice. For example, bad data could come from a malfunctioning sensor on a robot, or from someone spamming a search engine to try to rank high in search results. To reduce this risk, you need to monitor your system closely and promptly switch learning off (and possibly revert to a previously working state) if you detect a drop in performance. You may also want to monitor the input data and react to abnormal data (e.g., using an anomaly detection algorithm)


#### Instance-Based Learning
system learns by heart from training data and then generalizes to new cases based on a measure of similarity

#### Model based learning
Build model of examples and then use model to make predictions on new cases
Performance measure can be defined either as a [utility function] also called [fitness function] that measures how good your model is
[cost function] - measures how bad your model is

[Model selection] -choosing the type of model and fully specifying its architecture
Training a model means running an algorithm to find the model parameters that will make it best fit the training data (and hopefully make good predictions on new data)

[Inference]  - make predictions on new cases

### Challenges in ML

* Bad data
 - Not enough training data
 - non representative training data
   - if sample is too small, you will have [sampling noise] (i.e., nonrepresentative data as a result of chance)
   - Even very large samples can be nonrepresentative if the sampling method is flawed. This is called [sampling bias]
 - poor quality data (missing data, outliers)
 - irrelevant features. [feature engineering] includes -
    - feature selection
    - feature extraction e.g. dimension reduction
 
 
* Bad model
 - [overfitting] training data - it means that the model performs well on the training data, but it does not generalize well
 
 Overfitting happens when the model is too complex relative to the amount and noisiness of the training data. Here are possible solutions:
 - Simplify the model by selecting one with fewer parameters (e.g., a linear model rather than a high-degree polynomial model), by reducing the number of attributes in the training data, or by constraining the model.
 - Gather more training data.
 - Reduce the noise in the training data (e.g., fix data errors and remove outliers).


[Regularization] - Constraining a model to make it simpler and reduce the risk of overfitting
[degrees of freedom] in the model - no of parameters in the model
[hyperparameter] The amount of regularization to apply during learning can be controlled by a hyperparameter. A hyperparameter is a parameter of a learning algorithm (not of the model). As such, it is not affected by the learning algorithm itself; it must be set prior to training and remains constant during training.

if you set the regularization hyperparameter to a very large value, you will get an almost flat model (a slope close to zero); the learning algorithm will almost certainly not overfit the training data, but it will be less likely to find a good solution. Tuning hyperparameters is an important part of building a Machine Learning system.

[underfitting] - model is too simple for the data. Solution is to select a more powerful complex model,feed better features [ feature organising] or reduce constraints on the model [regularization hyperparameter]


#### Testing and validation

Data is split  into training and test ( typically 80/20 ratio )
error rate on validation data is called generalization error [out of sample error]. test data can give you an estimate of this error rate.
if test data error rate is low and generalization error rate is high - overfitting model


#### Hyperparameter Tuning and Model Selection

Because hyperparameter tuning on test set will only give good results for test dataset. this problem is overcome by [holdout validation]
: you simply hold out part of the training set to evaluate several candidate models and select the best one. The new held-out set is called the [validation set] (or sometimes the development set, or dev set). 
More specifically, you train multiple models with various hyperparameters on the reduced training set (i.e., the full training set minus the validation set), and you select the model that performs best on the validation set. After this holdout validation process, you train the best model on the full training set (including the validation set), and this gives you the final model. Lastly, you evaluate this final model on the test set to get an estimate of the generalization error

However, if the validation set is too small, then model evaluations will be imprecise: you may end up selecting a suboptimal model by mistake. Conversely, if the validation set is too large, then the remaining training set will be much smaller than the full training set. Why is this bad? Well, since the final model will be trained on the full training set, it is not ideal to compare candidate models trained on a much smaller training set. It would be like selecting the fastest sprinter to participate in a marathon. One way to solve this problem is to perform repeated [cross-validation], using many small validation sets. Each model is evaluated once per validation set after it is trained on the rest of the data. By averaging out all the evaluations of a model, you get a much more accurate measure of its performance. There is a drawback, however: the training time is multiplied by the number of validation sets.

#### Data Mismatch
validation set and the test set must be as representative as possible of the data you expect to use in production
[train-dev set] - After the model is trained (on the training set, not on the train-dev set), you can evaluate it on the train-dev set. If it performs well, then the model is not overfitting the training set. If it performs poorly on the validation set, the problem must be coming from the data mismatch. 
 
 Conversely, if the model performs poorly on the train-dev set, then it must have overfit the training set, so you should try to simplify or regularize the model, get more training data, and clean up the training data
