 Spam Email Detection and Topic Modeling using Apache Spark & Scala
 
This project implements a scalable and distributed system for spam email classification and topic modeling using Apache Spark and Scala.
It leverages Spark’s MLlib for machine learning and LDA (Latent Dirichlet Allocation) for uncovering hidden topics within email datasets.

 Features
Distributed Big Data Processing using Spark DataFrames and RDDs
Email Classification with:
-Logistic Regression
-Naive Bayes
-Random Forest
-Topic Modeling using Latent Dirichlet Allocation (LDA)
-Efficient Partitioning of a 45k+ email dataset for parallel computation
-Text Preprocessing: tokenization, stopword removal, TF-IDF transformation
-Performance Evaluation via Accuracy metrics
-Topic Extraction with top keywords per topic

Architecture Overview
Stage	Description
1. Data Preprocessing	Cleaned email text, removed URLs, numbers, and non-alphabetic characters
2. Feature Extraction	Used TF–IDF representation for classification
3. Classification Models	Trained Logistic Regression, Naive Bayes, and Random Forest models
4. Topic Modeling	Applied LDA to extract dominant topics from emails
5. Parallelism	Dataset repartitioned into 4 partitions to utilize Spark’s distributed processing
6. Evaluation	Compared model accuracies and generated topic distributions
   
 Technologies Used
Apache Spark (MLlib)
Scala
Spark SQL
LDA (Latent Dirichlet Allocation)
TF–IDF
HashingTF
StopWordsRemover
StringIndexer
MacOS (local environment)
