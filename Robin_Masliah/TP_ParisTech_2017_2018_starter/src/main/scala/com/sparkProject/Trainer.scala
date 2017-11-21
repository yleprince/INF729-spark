package com.sparkProject

import javassist.convert.Transformer

import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


object Trainer {


  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5  ROBIN E. MASLIAH
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      *       ./build_and_submit.sh Trainer
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/

    val df: DataFrame = spark
      .read
      .option("header", true)  // Use first line of all files as header
      .option("inferSchema", "true") // Try to infer the data types of each column
      .option("nullValue", "false")  // replace strings "false" (that indicates missing data) by null values
      .parquet("/home/robin/Documents/TP#Spark")

    df.show()
    println(df.count)

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("removed")

    /** TF-IDF **/

    val vectorizer = new CountVectorizer()
      .setInputCol("removed")
      .setOutputCol("vectorized")



    val tf_idf = new IDF()
      .setInputCol("vectorized")
      .setOutputCol("tfidf")

    val indexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("country2")
      .setOutputCol("country_indexed")
     // .fit(df)

    val indexer2 = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
     // .fit(df)



    /** VECTOR ASSEMBLER **/

    val vassembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa",
        "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")


    /** MODEL **/

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** PIPELINE **/
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, vectorizer, tf_idf, indexer, indexer2, vassembler, lr))


    val Array(training, test) = df.randomSplit(Array[Double](0.9, 0.1))

    /** TRAINING AND GRID-SEARCH **/


    // Fit the pipeline to training documents.

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(vectorizer.minDF, Array[Double](55, 75, 95))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")


    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.75)

    val model = trainValidationSplit.fit(training)

    val df_with_predictions = model.transform(test)

    println("f1_score = " + evaluator.setMetricName("f1").evaluate(df_with_predictions))

    df_with_predictions.groupBy("final_status", "predictions").count.show()

    model.write.overwrite().save("Model_LR_SPARK")

  }
}
