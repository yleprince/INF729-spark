package com.sparkProject
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer,RegexTokenizer,StringIndexer,IDF,StopWordsRemover,VectorAssembler}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression,LogisticRegressionModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder,TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import Array._

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
import spark.implicits._
spark.sparkContext.setLogLevel("ERROR")

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
   val df: DataFrame = spark
     .read
     .option("header", true)  // Use first line of all files as header
     .option("inferSchema", "true") // Try to infer the data types of each column
     .option("nullValue", "false")  // replace strings "false" (that indicates missing data) by null values
     .parquet("data/prepared_trainingset/")

    println(s"Total number of rows: ${df.count}")
    println(s"Number of columns ${df.columns.length}")

    /** TF-IDF **/
  val tokenizer = new RegexTokenizer()
    .setPattern("\\W+")
    .setGaps(true)
    .setInputCol("text")
    .setOutputCol("tokens")

  val stopWord = new StopWordsRemover()
    .setInputCol("tokens")
    .setOutputCol("tokensfiletered")

  val countVectorizer = new CountVectorizer()
    .setInputCol("tokensfiletered")
    .setOutputCol("rawFeatures")

    /** VECTOR ASSEMBLER **/

  val idf = new IDF()
    .setInputCol("rawFeatures")
    .setOutputCol("tfidf")


  val stringindexer_coun = new StringIndexer()
    .setInputCol("country2")
    .setOutputCol("country_indexed")


  val stringindexer_curr = new StringIndexer()
    .setInputCol("currency2")
    .setOutputCol("currency_indexed")

  val vecAssembler = new VectorAssembler()
    .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
    .setOutputCol("features")

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


    /** MODEL **/

  val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWord, countVectorizer,idf,stringindexer_coun,stringindexer_curr,vecAssembler,lr))

  val model = pipeline.fit(df)


  val splits = df.randomSplit(Array(0.9, 0.1), seed = 11L)
  val training = splits(0).cache()
  val test = splits(1)

    /** PIPELINE **/


    /** TRAINING AND GRID-SEARCH**/
  val lambda = Array(10e-8, 10e-6, 10e-4 ,10e-2)
  val mindf = Array(55,75,95)

  val paramGrid = new ParamGridBuilder()
        .addGrid(lr.regParam, (8 to 2 by -2).map(el => math.pow(10, -1 * el)))
        .addGrid(countVectorizer.minDF, (55.0 to 95.0 by 20.0))
        .build()


  val eval = new MulticlassClassificationEvaluator()
            .setLabelCol("final_status")
            .setPredictionCol("predictions")
            .setMetricName("f1")


  // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
  val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(eval)
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)

  // Run train validation split, and choose the best set of parameters.
  val bestModel = trainValidationSplit.fit(training)

  val df_pred = bestModel.transform(test)
  val f1_score = eval.evaluate(df_pred)
  println("\n\n\nf1 score on test set for the best model : " + (f1_score) + "\n\n\n")

  df_pred.groupBy("final_status", "predictions").count.show()

  val bestPipelineModel = bestModel.bestModel.asInstanceOf[PipelineModel]

  val stages = bestPipelineModel.stages

  println("Best parameters found on grid search :")

  val hashingStage = stages(2).asInstanceOf[CountVectorizerModel]
  println("\tminDF = " + hashingStage.getMinDF)

  val lrStage = stages(stages.length - 1).asInstanceOf[LogisticRegressionModel]
  println("\tregParam = " + lrStage.getRegParam)



  }
}
