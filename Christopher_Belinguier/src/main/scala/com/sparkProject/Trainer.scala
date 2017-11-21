package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
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
     .parquet("../prepared_trainingset")

    // df.show()
    df.printSchema()

    /** TF-IDF **/
    // a. Séparer les textes en mots (ou tokens) avec un tokenizer
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // b. Retirer les stop words pour ne pas encombrer le modèle avec des mots qui ne véhiculent pas de sens
    val stopWordSet = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("tokens_cl")

    // c. Calcul de la partie TF (term frequency)
    val vectorizer = new CountVectorizer()
      .setVocabSize(50)
      .setInputCol("tokens_cl")
      .setOutputCol("tf")

    // d. Calcul de la partie IDF
    val idf = new IDF()
      .setInputCol("tf")
      .setOutputCol("tfidf")


    /** Convertir les catégories en données numériques **/
    // e. Convertir les catégories "country2" en données numériques
    val indexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    // f. Convertir les catégories "currency2" en données numériques
    val indexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")


    /** Mettre les données sous une forme utilisable par Spark.ML **/
    // g. Assemble the features dans une seule colonne "features"
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf","days_campaign","hours_prepa","goal","country_indexed","currency_indexed"))
      .setOutputCol("features")

    // h. Modèle de classification : régression logistique
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

    // i. Créer le pipeline en assemblant les 8 stages définis précédemment, dans le bon ordre.
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordSet, vectorizer, idf, indexerCountry, indexerCurrency, assembler, lr))


    /** Entraînement et tuning du modèle **/
    /** Splitter les données en Training Set et Test Set **/
    // j. Créer un dataFrame nommé “training” et un autre nommé “test”
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1))

    /** Entraînement du classifieur et réglage des hyper-paramètres de l’algorithme **/
    // Créer une grille de valeurs à tester pour les hyper-paramètres
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(vectorizer.minDF, Array(55.0, 75.0, 95.0))
      .build()

    // Evaluateur de l'ensemble de validation du training set
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    // Séparer le training set en un ensemble de training (70%) et validation (30%)
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // Entrainer un modèle sur le training set et choisir les meilleurs parametres sur la grille
    val model = trainValidationSplit.fit(training)

    // l. Tester le modèle obtenu sur les données test
    // Mettre les résultats dans le dataFrame df_WithPredictions
    println(" ----- DONNEES TEST -----")
    val df_WithPredictions = model.transform(test)
      .select("features", "final_status", "predictions", "raw_predictions")

    // Afficher le f1-score du modèle sur les données de test
    println(" ----- F-MEASURE -----")
    val accuracy = evaluator.evaluate(df_WithPredictions)
    println("Logistic Regression Classifier Accuracy (F1-Measure): " + accuracy)

    // m. Afficher les predictions
    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    // Sauvegarder le modèle entraîné
    df_WithPredictions.write.mode(SaveMode.Overwrite).parquet("../predictions")

  }
}
