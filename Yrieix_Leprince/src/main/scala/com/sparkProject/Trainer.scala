package com.sparkProject


import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature.{IDF, StopWordsRemover}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


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
      *       - lire le fichier sauvegardé précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      * To launch:
      *   ./build_and_submit.sh Trainer
      ********************************************************************************/

    /** 1. CHARGER LE DATASET **/
    println("\n1.\tLoading csv file.")
    val df: DataFrame = spark
        .read
        .parquet("../../../../../../fundingData/prepared_trainingset")

    val nValuesToPrint = 10
    //df.printSchema()
    //df.show(nValuesToPrint)
    println("\t1.x)\t---\tFile loaded.")


    /** 2. TF-IDF **/
    println("\n2.\tTF-IDF.")

    // a) Tokenizer: text splitted in tokens
    println("\t2.a)\t---\tTokenizer.")
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // b) StopWordsRemover: meaningless words remover
    println("\t2.b)\t---\tStopWordsRemover.")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("stopWR")

    // c) Term Frequency - TF
    println("\t2.c)\t---\tTermFrequency.")
    val countVectorizer = new CountVectorizer()
      .setInputCol("stopWR")
      .setOutputCol("termFrequency")
      .setVocabSize(30)

    // d) Inverse Document Frequency - IDF
    println("\t2.d)\t---\tIDF.")
    val idf = new IDF()
      .setInputCol("termFrequency")
      .setOutputCol("tfidf")


    /** 3. Convertir​ ​ les​ ​ catégories​ ​ en​ ​ donnees​ ​ numeriques **/
    println("\n3.\tCategorical conversions.")

    // e) Convert “country2” (categorical) to numerical data.
    println("\t3.e)\t---\t'country2' --> 'country_indexed'.")
    val stringIndexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    // f) Convert “currency2” (categorical) to numerical data.
    println("\t3.f)\t---\t'currency2' --> 'currency_indexed'.")
    val stringIndexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")


    /** 4. VECTOR ASSEMBLER **/
    println("\n4.\tVector Assembler.")

    // g) Assembler les features dans une seule colonne 'features'.
    println("\t4.g)\t---\tfeatures --> vector assembler.")
    val vectorAssembler_features = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal",
        "country_indexed", "currency_indexed"))
      .setOutputCol("features")


    /** MODEL **/
    // h) Modele de classification : regression logistique
    println("\t4.h)\t---\tClassification model: Logistic Regression.")
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
    // i) Creation du pipeline avec les 8 etapes definies ci-dessus.
    println("\t4.i)\t---\tPipeline creation.")
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, countVectorizer, idf,
        stringIndexer_country, stringIndexer_currency, vectorAssembler_features, lr))


    /** TRAINING AND GRID-SEARCH **/
    println("\n5.\tTraining.")

    println("\t5.j)\t---\tTraining/test sets creation.")
//    val splits = df.randomSplit(Array(0.7, 0.3), seed = 0)
//    val training = splits(0).cache()
//    val test = splits(1)
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 0)

    println("\t5.k)\t---\tParam grid creation.")
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-9, 10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1))
      .addGrid(countVectorizer.minDF, Array(5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0))
      .build()

    println("\t5.l)\t---\tEvaluator creation.")
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    println("\t5.m)\t---\tValidation split creation.")
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    println("\t5.n)\t---\tFit model.")
    val my_model = trainValidationSplit.fit(training)

    println("\t5.o)\t---\tTransform test set.")
    val predicted_test = my_model
      .transform(test)
      .select("features", "final_status", "predictions", "raw_predictions")

    println("\t  \t---\tPredicted test set.")
    predicted_test.show(nValuesToPrint)

    println("\t5.p)\t---\tAccuracy.")
    val my_accuracy = evaluator
      .evaluate(predicted_test)
    println("\t  \t---\tAccuracy: " + my_accuracy)


    /** EXPORT RESULT **/
    println("\n6.\tExporting.")
    predicted_test.write.mode(SaveMode.Overwrite).parquet("../../../../../../fundingData/prediction")
    println("\t6.x)\t---\tFile exported.")
  }
}

