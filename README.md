**CatchPhish**
**Implementation**

**Dataset Pre-processing:**

The first step towards implementation is to finalize a good enough dataset to work on which consists of URLs of different kinds. Initially we find datasets specifically related to Twitter but most of them contain either Tweet based or machine activity based features with no ground truth to work on. Along with that, our research leads us to the fact that all such features are not necessarily easy to extract without the URL being clicked on. Moreover, our current findings lead us to a research paper  which detects malicious URLs on different OSNs on the basis vectors obtained from word embedding. Hence, we follow a similar approach and work on URLs in general and not just those URLs which are present in a tweet. 
We obtain a data set of 2019 , which is created to tackle the problem of malicious URLs on the Internet. It is acquired from various sources such as PhishTank etc. It contains records of benign and malicious URLs that can be used for analysis or building classifiers. The total number of unique URLs in the data set is 450176 out of which 77% are benign and 23% are malicious. The data set contains four columns representing the index, URL, label, and result. 
In order to use the data set for our work, we delete a few benign entries randomly to make the percentage of malicious and benign URLs equal i.e. 50% malicious and 50% benign to obtain better results. The final data set consists of 207081 entries with equal malicious and benign URLs. All the information from this data set is used for training and testing purposes. For training purposes, 80% of the data is used while testing is done on the rest of the 20% data. 

**Tokenizing the URLs:**

The first step of our implementation includes the process of tokenization of URLs, which is a necessary step of Natural Language Processing tasks. We use the basic tokenizer, which is present in NLTK – The Natural Language Toolkit. NLTK contains text processing libraries out of which we use the tokenize library to parse the URL into different tokens. We make an array to store the tokens after parsing. The tokens of each URL are appended one by one in the array which is later passed on to the vector model. Our tokenizer works in a way that it splits up the URL into different words by using an unsupervised algorithm to build a vector model which is later given as an input to LSTM. We use two modules of the tokenize library which include word_tokenize and sent_tokenize. 


**Word Embedding:**

The next step involved in the process is to create a vector model using the tokens that we obtain in the previous step. We do so by following the method of vectorization, also known as word embedding which is the process of converting words into numbers. Through this process, we extract information which is meaningful from a given word. We map words from vocabulary to a corresponding vector of real numbers and then find word semantics. After the words are converted as vectors, we use some techniques to find the similarity among the words. There are several word embedding approaches like BOW (Bag of Words), TF-IDF (Term Frequency — Inverse Document Frequency) Scheme and Word2Vec. The technique that we use for word embedding is Word2Vec. It works in such a way that the semantically similar vectors are near each other in N-dimensional space, where N is the dimensions of the vector. We import Word2Vec from the Gensim library to develop word embeddings by training our model on a custom corpus by the skip-gram algorithm. We use nltk.sent_tokenize and nltk.word_tokenize utilities for training of the model.

**Long Short-Term Memory:**

The final step is to feed the vectors into a neural network. We use LSTM as our classification model which is a tweaked version of recurrent neural network. LSTMs can make little modifications by multiplications and additions depending upon the need as they can selectively remember or forget things. After the creation of vectors, we work on feeding those vectors to the LSTM for training. We prepare the data in such a format that for the prediction of ’E’ in ’BLUE’, the input given is the first three letters in an array and the outcome of that input is the last letter. To define the LSTM model, we use the sequential model. The first layer is embedding which takes the vocabulary size as input dimension, embedding size as output dimension and the pre-trained weights as the weights. We then add an LSTM layer to make sure that the problem of receiving of data in a scattered manner by the next LSTM layer can be avoided. We use the dropout layer after every LSTM layer to avoid over-fitting. Next, we use a fully connected layer i.e. dense layer and the activation function which we use with it is ‘relu’. Then, we apply another dense layer with 1 unit. Lastly, we use Sigmoid function the output of which is basically a vector. The vector has values 0 and 1 where 0 depicts that the value should be forgotten whereas 1 means that the value should be remembered. We then compile our model by setting the loss to binary_crossentropy. The optimizer is set as Adam and the metrics is set to accuracy. After that, we fit our model and run 10 epochs on it for training which gives us an accuracy of 98.3%. 

**Implementing Conventional Models**
We train and test three different models on the same data set using the following techniques and obtain different accuracies with each one of them. 
1.	Random Forest
2.	Naive Bayes
3.	Multilayer Perceptron
The input which is fed to each of them is the same as that of LSTM. However, the vectors are treated as attributes in case of these three models. This makes it easy to compare our proposed approach model to the conventional models having the same input.

**Results and Discussion  
Overview:**

We train different classification models as mentioned above. We present the results of these models using different evaluation measures which include accuracy, f1-score, precision and recall. The results depict that LSTM achieves the highest accuracy i.e. 99.2% in order to detect the maliciousness of URLs with a loss of only 0.04. The remaining models also give good accuracies which are 99%, 98%, and 95% for Random Forest, Multilayer Perceptron, and Naïve Bayes respectively, but are lesser than that of the proposed model. 
5.1 Results:
The results obtained are presented in Table 2 below.
Table 2: Results of Classification Techniques
Algorithm	Accuracy	F1-score	Precision	Recall
LSTM	0.992	0.992	0.992	0.993
Random Forest	0.990	0.985	0.976	0.990
MLP	0.981	0.981	0.970	0.993
Naïve Bayes	0.953	0.953	0.924	0.991

**Result Plots:**

ROC Curve:


![image](https://user-images.githubusercontent.com/55654110/149082806-a0b358c5-792e-4995-93cf-9ab98b4c817a.png)
![image](https://user-images.githubusercontent.com/55654110/149082884-41a93e25-bfd2-427d-9fc5-068f5578dd07.png)

Confusion Matrix:

![image](https://user-images.githubusercontent.com/55654110/149082934-59c496ab-1787-4bdd-9604-f7c1136b975a.png)
![image](https://user-images.githubusercontent.com/55654110/149082950-28ed618f-a6a4-4708-8040-e5d6293e652c.png)

**Conclusion:**

We have come to the conclusion, after analysing the results, that using word embeddings is the best approach to go with as this yields fast and accurate results. Our results indicate that the feature vectors obtained seem to have enough predictive power to be used in practice. Along with that, we identified that the best technique for this problem is LSTM. We also provide a fast and light weighted operation, as the features used in our work can easily be collected easily within a short period, without having to be clicked and hence that gives us an advantage over a model that collect features specifically from tweets or are machine activity-based which need to be clicked at any cost in order to be collected. The comparison of our approach with highly tuned conventional models has also been done and found our approach to be giving the highest results. 

