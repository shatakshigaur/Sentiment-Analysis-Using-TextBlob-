# Sentiment-Analysis-Using-TextBlob-
1. Sentiment Analysis using TextBlob
Objective: To classify the sentiment of text data as positive, negative, or neutral using the TextBlob library.
Key Steps:

TextBlob calculates the polarity score of a text, which ranges from -1 (most negative) to 1 (most positive).
Based on the polarity, the text is classified into three categories:
Positive: Polarity > 0
Negative: Polarity < 0
Neutral: Polarity = 0
The results are stored in a new column (textblob_sentiment), which contains the predicted sentiment labels.

Performance Evaluation:
The predictions are compared against the actual sentiment labels in the dataset using metrics like accuracy, precision, recall, and F1 score.
This analysis helps evaluate TextBlob’s effectiveness as a sentiment analysis tool on the given text data.


2. TF-IDF Vectorization and Logistic Regression for Sentiment Classification
Objective: To classify text data using a supervised machine learning model (Logistic Regression) based on TF-IDF features.
Key Steps:

TF-IDF Vectorization (TfidfVectorizer):
Converts text into numerical feature vectors by assigning a score (TF-IDF) to each word.
This transformation captures word importance based on term frequency and inverse document frequency, resulting in a sparse matrix of text features.

Label Encoding:
Manually labeled or rule-based sentiment labels (positive, negative, and neutral) are converted into numerical form (1, -1, and 0 respectively).
This step prepares the target variable (y) for model training.
Train-Test Split:
The dataset is split into a training set (70%) and a test set (30%) to evaluate model performance on unseen data.

Logistic Regression Classifier:
A Logistic Regression model is initialized and trained using the training set.
The model learns the relationship between the text features (TF-IDF vectors) and the sentiment labels.

Prediction and Evaluation:
Predictions are made on the test set, and accuracy, precision, recall, and F1 score are computed.
This model evaluation helps to understand the classifier’s performance.


3. Topic Modeling using Latent Dirichlet Allocation (LDA)
Objective: To uncover latent topics in a collection of text data using LDA.
Key Steps:

Text Preprocessing:
Each document is split into a list of individual words.

Dictionary and Corpus Creation:
corpora.Dictionary: Creates a dictionary mapping each word in the dataset to a unique identifier.
Corpus: Converts each document into a bag-of-words (BoW) format, representing each document as a list of word frequency tuples.

LDA Model Training:
The LDA model is trained on the corpus to find num_topics (e.g., 5) distinct topics.
Each topic is characterized by a distribution over words, and each document is represented as a mixture of these topics.

Print Topics:
The top words for each topic are printed, providing insight into the semantic structure of the text.


4. Text Classification Using Logistic Regression
Objective: To implement a complete machine learning pipeline for text classification using TF-IDF features and a Logistic Regression classifier.
Pipeline Overview:

TF-IDF Vectorization:
Transforms text into numerical form using TF-IDF, capturing the importance of each word.

Model Initialization and Training:
A Logistic Regression model is trained using the vectorized text.

Prediction:
The model predicts the sentiment labels for the test data.

Performance Evaluation:
Accuracy: The percentage of correctly predicted labels.
Precision, Recall, and F1 Score: These metrics provide a detailed evaluation of how well the model performs for each class, especially in the presence of class imbalance.
Concepts and Techniques Covered:

Text Preprocessing:
Cleaning and preparing raw text data for analysis, such as removing special characters, stop words, and performing tokenization.

Sentiment Analysis:
Using both rule-based (TextBlob) and supervised (Logistic Regression) techniques for sentiment classification.

TF-IDF Vectorization:
A common method for feature extraction in text classification tasks. Captures word importance in a numerical form.

Machine Learning Pipeline:
Implementing a complete supervised learning pipeline, covering train-test split, model training, prediction, and evaluation.

Topic Modeling:
Uncovering hidden thematic structures in a large corpus using Latent Dirichlet Allocation (LDA).

Performance Metrics:
Detailed evaluation using accuracy, precision, recall, and F1 score, giving a comprehensive view of model performance.

Typical Use Cases for Each Technique:

TextBlob for Sentiment Analysis:
Suitable for quick, rule-based sentiment analysis.
Best for exploratory analysis and initial sentiment evaluation.

Logistic Regression with TF-IDF Features:
Ideal for supervised text classification tasks.
Works well for sentiment analysis, spam detection, and other binary/multiclass text classification problems.

LDA for Topic Modeling:
Used to discover latent topics in large corpora.
Effective for summarizing large-scale text data, understanding themes, and organizing documents based on topics.
This summary gives a detailed walkthrough of each code segment and its significance in solving typical NLP tasks using both rule-based and machine learning approaches.
