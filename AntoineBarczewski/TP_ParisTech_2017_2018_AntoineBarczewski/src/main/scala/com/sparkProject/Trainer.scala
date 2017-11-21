package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession


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
   import spark.implicits._
   val df = spark.read.parquet("/cal/homes/mbarczewski/INF729/prepared_trainingset")
   df.show(5)


    /** TF-IDF **/
    import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover
      , CountVectorizer, CountVectorizerModel, IDF}

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    val cvModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")

    val idf = new IDF()
    .setInputCol("rawFeatures")
    .setOutputCol("tfidf")

    /** string indexer for country**/
    import org.apache.spark.ml.feature.StringIndexer

    val indexerCountry = new StringIndexer()
    .setInputCol("country2")
    .setOutputCol("country_indexed")

    val indexerCurrency = new StringIndexer()
    .setInputCol("currency2")
    .setOutputCol("currency_indexed")

    /** VECTOR ASSEMBLER **/
    import org.apache.spark.ml.feature.VectorAssembler
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal",
        "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    /** MODEL **/
    import org.apache.spark.ml.classification.LogisticRegression
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
    import org.apache.spark.ml.Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, idf,
        indexerCountry, indexerCurrency, assembler, lr))

    /** TRAINING AND GRID-SEARCH **/
    /** Training and test sets**/
    import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 12345)


    /** grid-search**/
    val paramGrid = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2))
    .addGrid(cvModel.minDF, Array(55.0, 75.0, 95.0))
    .build()

    /**evaluator**/
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    /**train-validation split**/
    val trainValidationSplit = new TrainValidationSplit()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    // 70% of the data will be used for training and the remaining 30% for validation.
    .setTrainRatio(0.7)

    /** fit model **/
    val model = trainValidationSplit.fit(training)

    /** EVALUATE **/
    val df_WithPredictions = model.transform(test)
    val score = evaluator.evaluate(df_WithPredictions)

    println("F1-score = " + score)
    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    /** SAVE **/
    model.write.overwrite().save("/cal/homes/mbarczewski/INF729/spark-lr-model")


  }
}
