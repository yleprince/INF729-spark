package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}

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
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/
   val df: DataFrame = spark.read.parquet("/Users/quentin/Desktop/TP_ParisTech_2017_2018_starter/funding-successful-projects-on-kickstarter/train_cleaned")

    val indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("countryIndex")

    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currencyIndex")

    val encoder = new OneHotEncoder()
      .setInputCol(indexer.getOutputCol)
      .setOutputCol("country_indexed")

    val encoder2 = new OneHotEncoder()
      .setInputCol(indexer2.getOutputCol)
      .setOutputCol("currency_indexed")

    /** TF-IDF **/
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("words_filtered")

    val cvm = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("words_features")


    /** VECTOR ASSEMBLER **/

    val assembler = new VectorAssembler()
      .setInputCols(Array("days_campaign", "words_features", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
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
      .setStages(Array(tokenizer, remover, cvm, indexer, encoder, indexer2, encoder2, assembler, lr))


    /** Split Training Set **/
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 123)

    // Mise en cache pour accélérer les accès mémoires
    val training_cache = training.cache()


    /** TRAINING AND GRID-SEARCH **/
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(cvm.minDF, Array[Double](55.0,75.0,95.0))
      .build()

    val multi_evaluator =  new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")


    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(multi_evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)



    // Run cross-validation, and choose the best set of parameters.
    val model = trainValidationSplit.fit(training_cache)

    val df_with_predictions  = model.transform(test)

    println("f1 "+ multi_evaluator.evaluate(df_with_predictions))

    df_with_predictions.groupBy("final_status", "predictions").count.show()

  }
}
