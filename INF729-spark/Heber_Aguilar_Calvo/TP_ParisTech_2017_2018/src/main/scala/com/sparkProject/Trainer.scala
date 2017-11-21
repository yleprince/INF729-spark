package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{TrainValidationSplit,ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}


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

   /** 1. CHARGER LE DATASET **/
   val df: DataFrame = spark
     .read
     .option("header", true)  // Use first line of all files as header
     .option("inferSchema", "true") // Try to infer the data types of each column
     .option("nullValue", "false")  // replace strings "false" (that indicates missing data) by null values
     .parquet("/home/heber/TP_ParisTech_2017_2018/data/prepared_trainingset")

    //df.show()

   /**2. TF-IDF **/
   /*a. 1er Stage: Separer les textes en mots (tokens) avec un tokenizer.​ Construire​ ​ premier​ ​ Stage​ ​ du​ ​ pipeline​ ​ en​ ​ faisant:*/
   val tokenizer = new RegexTokenizer()
     .setPattern("\\W+")
     .setGaps(true)
     .setInputCol("text")
     .setOutputCol("tokens")
    //var wordsData = tokenizer.transform(df)
    //wordsData.show()

    /*b. 2me Stage: Retirer stop words pour ne pas encombrer le modèle avec des mots sans sens. Créer le 2ème stage avec la classe
    StopWordsRemover*/
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("tokens_b")
    //var wordsData = remover.transform(wordsData)
    //wordsData.show()

    /*c. 3me Stage: ​La​ partie​ TF​ de​ ​TF-IDF​ ​est​ ​faite​ ​avec​ ​la​ ​classe​ ​CountVectorizer*/
    val cvModel = new CountVectorizer()
      .setInputCol("tokens_b")
      .setOutputCol("tf")
      //.fit(df)

    //var wordsData = cvModel.transform(wordsData)
    //wordsData.show()

    /*d.4me Stage: Trouvez la partie IDF. On veut ecrire l'output de cette etape dans une colonne "tfidf"*/
    val idf = new IDF()
      .setInputCol("tf")
      .setOutputCol("tfidf")

    /**3. CATEGORIES TO NUMERIC DATA**/

    /*e. 5me Stage:Convertir la variable cat.“country2” en donnees numeriques, dans une​ ​ colonne​ ​ "country_indexed"*/
    val country_indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    /*f. 6me Stage: Convertir la variable catégorielle “currency2” en donnees numerique, dans la  colonne​ ​ "currency_indexed"*/
    val currency_indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    /** VECTOR ASSEMBLER **/
    /*g. 7me Stage: Assembler "tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed",​​ "currency_indexed"​ ​
      dans​ ​ une​ ​ seule​ ​ colonne​ ​ “features”*/
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf","days_campaign","hours_prepa","goal","country_indexed","currency_indexed"))
      .setOutputCol("features")

    /** MODEL **/
    /*h. 8me Stage: Modele de classification. Il s’agit d’une regression logistique definie de​ ​ la​ ​ façon​ ​ suivante:*/
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
    /**Enfin, creer le pipeline en assemblant les 8 stages dans le​ ​ bon​ ​ ordre.**/
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, idf, country_indexer, currency_indexer,assembler,lr))
    //val model = pipeline.fit(df) # Fit pipeline with all the data set

    /** TRAINING AND GRID-SEARCH **/
    /*j. Creer un dataFrame “training” et un autre “test” a partir du dataFrame charge initialement de façon
     à le séparer en training et test sets dans les​ ​ proportions​ ​ 90%,​ ​ 10%​ ​ respectivement*/
    val splits = df.randomSplit(Array(0.9, 0.1))
    val (training, test) = (splits(0), splits(1))

    /*k. Preparer la grid-search pour satisfaire les conditions explicitees,
    puis lancer​ la​ ​ grid-search​ sur​ le​ ​ dataset​ “training”​ ​ prepare​ ​ precedemment.*/
    val paramGrid = new ParamGridBuilder()
      .addGrid(cvModel.minDF, Array[Double](55,75,95))
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .build()

    /*Set the metric to measure how well a fitted Model does on held-out test data*/
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    /*On va utiliser Validation Croissee pour determiner les meilleurs parametres du modele*/
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
    //val cv = new CrossValidator()
    //  .setEstimator(pipeline)
    //  .setEvaluator(evaluator)
    //  .setEstimatorParamMaps(paramGrid)
    //  .setNumFolds(3)  // Use 3+ in practice

    val cvm = trainValidationSplit.fit(training)

    /*l. Appliquer le meilleur modele trouvé avec la grid-search aux données test. Mettre les resultats dans le
    dataFrame ​ df_WithPredictions. Afficher le f1-score du modele​ ​ sur​ ​ les​ ​ donnees​ ​ de​ ​ test.*/
    val df_with_predictions = cvm.transform(test)
    val score_f1 = evaluator.setMetricName("f1").evaluate(df_with_predictions)
    println("Le score f1 est egale a " + score_f1)

    /*m. Afficher​ ​df_WithPredictions.groupBy(​"final_status"​, "predictions").count.show()*/
    df_with_predictions.groupBy("final_status", "predictions").count.show()

    cvm.write.overwrite().save("Logistic_Reg_Model_TP_Spark")

  }
}
