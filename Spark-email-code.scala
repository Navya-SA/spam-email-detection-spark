Load dataset:
val dfClean = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .option("multiLine", "true")      // important to read full sentences with line breaks
  .option("escape", "\"")           // handle quotes correctly
  .csv("hdfs://localhost:9000/user/navya/cleaned_email_dataset/")

dfClean.printSchema()
dfClean.show(5, truncate = false)   // show full text
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val dfPartitioned = dfClean.cache().repartition(4)  // 4 partitions
println(s"Total rows: ${dfPartitioned.count()}")
// Convert label to numeric
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("labelIndex")
  .setHandleInvalid("skip")   // skip any unexpected nulls

// Tokenize text
val tokenizer = new Tokenizer()
  .setInputCol("text")
  .setOutputCol("words")

// Remove stop words
val remover = new StopWordsRemover()
  .setInputCol("words")
  .setOutputCol("filtered")

// TF-IDF features
val hashingTF = new HashingTF()
  .setInputCol("filtered")
  .setOutputCol("rawFeatures")
  .setNumFeatures(20000)

val idf = new IDF()
  .setInputCol("rawFeatures")
  .setOutputCol("features")
val Array(train, test) = dfPartitioned.randomSplit(Array(0.8, 0.2), seed = 42)
println(s"Training rows: ${train.count()}, Test rows: ${test.count()}")
val lr = new LogisticRegression()
  .setLabelCol("labelIndex")
  .setFeaturesCol("features")

val lrPipeline = new Pipeline()
  .setStages(Array(labelIndexer, tokenizer, remover, hashingTF, idf, lr))

val lrModel = lrPipeline.fit(train)
val lrPred = lrModel.transform(test)

// Evaluate
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("labelIndex")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

println(s"Logistic Regression Accuracy = ${evaluator.evaluate(lrPred)}")
val nb = new NaiveBayes()
  .setLabelCol("labelIndex")
  .setFeaturesCol("features")

val nbPipeline = new Pipeline()
  .setStages(Array(labelIndexer, tokenizer, remover, hashingTF, idf, nb))

val nbModel = nbPipeline.fit(train)
val nbPred = nbModel.transform(test)

println(s"Naive Bayes Accuracy = ${evaluator.evaluate(nbPred)}")
val rf = new RandomForestClassifier()
  .setLabelCol("labelIndex")
  .setFeaturesCol("features")
  .setNumTrees(50)

val rfPipeline = new Pipeline()
  .setStages(Array(labelIndexer, tokenizer, remover, hashingTF, idf, rf))

val rfModel = rfPipeline.fit(train)
val rfPred = rfModel.transform(test)

println(s"Random Forest Accuracy = ${evaluator.evaluate(rfPred)}")
val lrAcc = evaluator.evaluate(lrPred)
val nbAcc = evaluator.evaluate(nbPred)
val rfAcc = evaluator.evaluate(rfPred)

println(s"Accuracy: Logistic Regression = $lrAcc | Naive Bayes = $nbAcc | Random Forest = $rfAcc”)

// First, rename predictions so we can join
val lrPredRenamed = lrPred.select($"text", $"labelIndex".alias("actualLabel"), $"prediction".alias("lrPrediction"))
val nbPredRenamed = nbPred.select($"text", $"prediction".alias("nbPrediction"))
val rfPredRenamed = rfPred.select($"text", $"prediction".alias("rfPrediction"))

// Join all three predictions on 'text'
val joinedPreds = lrPredRenamed
  .join(nbPredRenamed, Seq("text"))
  .join(rfPredRenamed, Seq(“text"))

import org.apache.spark.sql.functions.{rand, expr}

// Randomly select 10 rows with truncated text
joinedPreds.orderBy(rand())
  .limit(10)
  .select(
    expr("substring(text, 1, 100) as text"), // first 100 chars of text
    $"actualLabel",
    $"lrPrediction",
    $"nbPrediction",
    $"rfPrediction"
  )
  .show(truncate=false)  // truncate=false keeps our truncated text fully visible


import org.apache.spark.ml.clustering.{LDA, LDAModel}
import org.apache.spark.sql.functions.{col, udf, expr, count, lower, regexp_replace}
import org.apache.spark.sql.types.StringType
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, HashingTF}

// Enhanced preprocessing: Clean and normalize text
val cleanedDf = dfClean.withColumn("cleanText", 
  lower(regexp_replace(col("text"), 
    "[^a-zA-Z\\s]|\\b\\d+\\b|https?://\\S+|www\\.\\S+|\\b[xX]+\\b|\\benron\\b|\\bnbsp\\b|\\biso\\b|\\b8859\\b|\\bfw\\b|\\bfrom\\b|\\bwrote\\b", " ")))

val cleaner = new RegexTokenizer()
  .setInputCol("cleanText")
  .setOutputCol("rawWords")
  .setPattern("\\w{3,}") // Keep words with 3+ characters
  .setToLowercase(true)

val remover = new StopWordsRemover()
  .setInputCol("rawWords")
  .setOutputCol("filtered")
  .setStopWords(StopWordsRemover.loadDefaultStopWords("english") ++ 
    Array("re", "fwd", "cc", "subject", "http", "com", "net", "org", "date", "wrote", "from", "tap", "gas", "daily", "sir", "madam", "dear", "click", "font", "family"))

// Optimize feature space
val hashingTF = new HashingTF()
  .setInputCol("filtered")
  .setOutputCol("rawFeatures")
  .setNumFeatures(8000) // Further reduced to optimize performance

// Define topic names (refined based on expected categories)
val topicNames = Array("Promotions", "Ads", "Personal", "Spam", "Work", "Meetings", "Newsletters", "Other", "Marketing", "Technical")

// Apply LDA for topic modeling
val lda = new LDA()
  .setK(12) // Increased to 12 for more granularity
  .setMaxIter(100) // More iterations for convergence
  .setOptimizer("online")
  .setDocConcentration(3.0) // Alpha: Encourage diverse topic assignments
  .setTopicConcentration(0.01) // Beta: Sharper word distributions
  .setSeed(42) // For reproducibility
  .setFeaturesCol("rawFeatures")
  .setTopicDistributionCol("topicDistribution")

// Create a pipeline for LDA
val ldaPipeline = new Pipeline()
  .setStages(Array(cleaner, remover, hashingTF, lda))

// Fit LDA model on the cleaned dataset
val ldaModel = ldaPipeline.fit(cleanedDf)

// Transform the test set
val testCleaned = test.withColumn("cleanText", 
  lower(regexp_replace(col("text"), 
    "[^a-zA-Z\\s]|\\b\\d+\\b|https?://\\S+|www\\.\\S+|\\b[xX]+\\b|\\benron\\b|\\bnbsp\\b|\\biso\\b|\\b8859\\b|\\bfw\\b|\\bfrom\\b|\\bwrote\\b", " ")))
val ldaPred = ldaModel.transform(testCleaned)

// UDF to extract the most likely topic index
val getTopTopic = udf((topicDist: Vector) => {
  topicDist.toArray.zipWithIndex.maxBy(_._1)._2
})

// UDF to map topic index to topic name
val getTopicName = udf((topicIdx: Int) => topicNames(topicIdx % topicNames.length)) // Handle extra topics

// Add topic index and topic name columns
val ldaPredWithTopic = ldaPred
  .withColumn("topicIndex", getTopTopic(col("topicDistribution")))
  .withColumn("topicName", getTopicName(col("topicIndex")))

// Select relevant columns for joining
val ldaPredRenamed = ldaPredWithTopic
  .select($"text", $"topicName")

// Join with existing predictions
val finalPreds = joinedPreds
  .join(ldaPredRenamed, Seq("text"))

// Display 10 random rows
finalPreds.orderBy(rand())
  .limit(10)
  .select(
    expr("substring(text, 1, 100) as text"),
    $"actualLabel",
    $"lrPrediction",
    $"nbPrediction",
    $"rfPrediction",
    $"topicName"
  )
  .show(10, false)


