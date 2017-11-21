package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{IDFModel, IDF}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.regression.LabeledPoint

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

    /** 1 - CHARGEMENT DU DATAFRAME **/

    val df = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("nullValue", "false")
      .parquet("/Users/Marion/Documents/MSBigData/Cours/Période1/INF729/TP_ParisTech_2017_2018_starter/prepared_trainingset")

    df.show()


    /** 2 - UTILISATION DE L'ALGORITHME TF-IDF POUR LA GESTION DES DONNEES TEXTUELLES **/

    /** A) PREMIER STAGE : SEPARATION DES TEXTES EN MOTS (OU TOKENS) **/
    val tokenizer = new RegexTokenizer()
      .setPattern( "\\W+" )
      .setGaps( true )
      .setInputCol( "text" )
      .setOutputCol( "tokens" )

    //val regexTokenized = tokenizer.transform(df)
    //regexTokenized.select("tokens").show()
    //println(s"Total number of rows : ${regexTokenized.count}")


    /** B) DEUXIEME STAGE : ON ENLEVE LES STOP WORDS **/
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered_SW")

    //val removed = remover.transform(regexTokenized)
    //removed.select("filtered_SW").show()
    //println(s"Total number of rows : ${removed.count}")


    /** C) TROISIEME STAGE : UTILISATION DE COUNTVECTORIZER (PARTIE TF DE TF-IDF)**/
    val count_vect: CountVectorizer/*Model*/ = new CountVectorizer()
      .setInputCol("filtered_SW")
      .setOutputCol("tf-count")//.fit(removed)

    //val TF = count_vect.transform(removed)
    //TF.show()
    //println(s"Total number of rows : ${TF.count}")


    /** D) QUATRIEME STAGE : RECHERCHE DE LA PARTIE IDF ET ECRITURE DE L'OUTPUT DANS UNE COLONNE "TFIDF"**/
    val idf: IDF/*Model*/ = new IDF()
      .setInputCol("tf-count")
      .setOutputCol("TFIDF")//.fit(TF)

    //val IDF2 = idf.transform(TF)
    //IDF2.select("TFIDF").show()
    //println(s"Total number of rows : ${IDF2.count}")


    /** 3 -  CONVERSION DES CATEGORIES EN DONNEES NUMERIQUES **/

    /** E) CINQUIEME STAGE : CONVERSION COUNTRY2 EN DONNEES NUMERIQUES **/
    val indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country2_indexed")

    //val indexer_bis = indexer.fit(IDF2).transform(IDF2)
    //indexer_bis.show()
    //println(s"Total number of rows : ${indexer_bis.count}")



    /** F) SIXIEME STAGE : CONVERSION CURRENCY EN DONNEES NUMERIQUES **/
    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency2_indexed")

    //val indexer2_bis = indexer2.fit(indexer_bis).transform(indexer_bis)
    //indexer2_bis.show()
    //println(s"Total number of rows : ${indexer2_bis.count}")


    /** 4 - CONVERSION DES DONNEES SOUS UNE FORME UTILISABLE PAR SPARK ML **/

    /** G) SEPTIEME STAGE : VECTOR ASSEMBLER **/
    val vector_assembler = new VectorAssembler()
      .setInputCols(Array("TFIDF", "days_campaign", "hours_prepa", "goal", "country2_indexed", "currency2_indexed"))
      .setOutputCol("Features")

    //val output = vector_assembler.transform(indexer2_bis)
    //output.select("Features").show()
    //println(s"Total number of rows : ${output.count}")

    /** H) HUITIEME STAGE : REGRESSION LOGISTIQUE **/
    val lr = new LogisticRegression()
      .setElasticNetParam( 0.0 )
      .setFitIntercept(true)
      .setFeaturesCol( "Features" )
      .setLabelCol( "final_status" )
      .setStandardization( true )
      .setPredictionCol( "predictions" )
      .setRawPredictionCol( "raw_predictions" )
      .setThresholds( Array ( 0.7 , 0.3 ))
      .setTol( 1.0e-6 )
      .setMaxIter( 300 )


    /** I) PIPELINE **/
    val Pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, count_vect, idf, indexer, indexer2, vector_assembler, lr))


    /** 5 - ENTRAINEMENT ET TUNING DU MODELE **/

    /** J) SPLIT DES DONNEES EN TRAINING SET ET TEST SET **/
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1))


    /** K) PREPARATION DU GRID-SEARCH POUR SATISFAIRE LES CONDITIONS **/
    val Grid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.00000001, 0.000001, 0.0001, 0.01))
      .addGrid(count_vect.minDF, Array(55.0, 75.0, 95.0))
      .build()

    /** PREPARATION DU DATASET TRAINING ET LANCEMENT DU GRID-SEARCH **/
    val f1_Score= new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")


    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(f1_Score)
      .setEstimator(Pipeline)
      .setEstimatorParamMaps(Grid)
      .setTrainRatio(0.7)


    /** L) APPLICATION DU MEILLEUR MODELE AUX DONNEES TEST**/
    val best_model = trainValidationSplit.fit(training)
    val df_WithPredictions = best_model.transform(test)

    print("f1 score sur les données de test: " + f1_Score.evaluate(df_WithPredictions))

    /** M) AFFICHAGE DES PREDICTIONS **/
    df_WithPredictions.groupBy( "final_status" , "predictions" ).count.show()
  }
}
