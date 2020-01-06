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

* Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do (instance-based versus model-based learning)

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



