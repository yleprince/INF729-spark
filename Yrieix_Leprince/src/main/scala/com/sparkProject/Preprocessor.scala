package com.sparkProject

import org.apache.spark.{SparkConf, sql}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}



object Preprocessor {

  def main(args: Array[String]): Unit = {

    /* Permet de paralléliser le code : */
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._


    /*******************************************************************************
      *
      *       TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      *
      * ./build_and_submit.sh Preprocessor
      ********************************************************************************/

    /** 1 - CHARGEMENT DES DONNEES **/


    /** a) CSV -> DataFrame
      * Charger le fichier csv dans un dataFrame.
      * La première ligne du fichier donne le nom de chaque colonne,
      * on veut que cette ligne soit utilisée pour nommer les colonnes
      * du dataFrame. On veut également que les “false” soient importés
      * comme des valeurs nulles.
      */



    println("\nLoading csv file.")

    val trainData: DataFrame = spark
      .read
      .option("header", "true")  // first line = header
      .option("inferSchema", "true") //
      .option("nullValue", "false")
      .csv("/home/bud/Documents/s1/spark2/fundingData/train.csv")
    println("File loaded.\n")

    /** b) Nb Lignes et Colonnes
      * Afficher le nombre de lignes et le nombre de colonnes dans le dataFrame.
      */
    val nbLines = trainData.count()
    val nbColumns = trainData.columns.length

    //println("\n" + "nbLines " + nbLines.toString)
    //println("nbColumns " + nbColumns.toString + "\n")


    /** c) Utilisation de show
      * Afficher le dataFrame sous forme de table.
      */
   // trainData.show()
   // trainData.select("name", "launched_at").show()


    /** d) Utilisation de printSchema
      * Afficher le schéma du dataFrame (nom des colonnes et le type des données contenues dans chacune d’elles).
      */
    trainData.printSchema()

    /** e) Changer le type
      * Assigner le type “Int” aux colonnes qui vous semblent contenir des entiers.
      */

//    trainData.select("goal").take(5).foreach(println)

    val featureDf = trainData
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline", $"deadline".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    featureDf.printSchema()
    println("Int data casted.\n")



    /** 2 - CLEANING **/

    /** a) Description statistique
      * Afficher une description statistique des colonnes de type Int (avec .describe().show )
      */
    println("\nStatistical visualization:")
    featureDf.select("goal", "backers_count", "final_status").describe().show
    println("End stat visualization\n")


    /** b) Cleaning potentiels des colonnes
      * Observer les autres colonnes, et proposer des cleanings à faire sur les données:
      * faites des groupBy count, des show, des dropDuplicates.
      * Quels cleaning faire pour chaque colonne ?
      * Y a-t-il des colonnes inutiles ?
      * Comment traiter les valeurs manquantes ?
      * Des “fuites du futur” ???
      */

    val nValueToShow = 10
    println("\nObservations of the data:")
    featureDf.select("project_id", "name", "desc", "goal",
      "keywords", "disable_communication", "country").describe().show(nValueToShow)

    featureDf.groupBy("disable_communication").count.orderBy($"count".desc).show(nValueToShow)
    featureDf.groupBy("country").count.orderBy($"count".desc).show(nValueToShow)
    featureDf.groupBy("currency").count.orderBy($"count".desc).show(nValueToShow)
    featureDf.select("deadline").dropDuplicates.show(nValueToShow)
    featureDf.groupBy("state_changed_at").count.orderBy($"count".desc).show(nValueToShow)
    featureDf.groupBy("backers_count").count.orderBy($"count".desc).show(nValueToShow)
    featureDf.select("goal", "final_status").show(nValueToShow)
    featureDf.groupBy("country", "currency").count.orderBy($"count".desc).show(nValueToShow)
    println("Ending data observations.\n")


   // featureDf.select("currency", "deadline", "state_changed_at", "created_at",
   //   "launched_at", "backers_count", "final_status").describe().show

    /** c) Drop Disable_communication
      * enlever la colonne "disable_communication".
      * cette colonne est très largement majoritairement à "false",
      * il y a 311 "true" (négligeable) le reste est non-identifié.
      */

    println("\nDropping column disable_communication:")
    println("nbColumns " + featureDf.columns.size.toString )
    val featureDF_drop = featureDf.drop(featureDf.col("disable_communication"))
    println("nbColumns " + featureDF_drop.columns.size.toString)
    println("Column disable_communication dropped.\n")

    /** d) Retirer les donnees du futur
      * Les fuites du futur: dans les datasets construits a posteriori des évènements,
      * il arrive que des données ne pouvant être connues qu'après la résolution de
      * chaque évènement soient insérées dans le dataset. On a des fuites depuis le
      * futur ! Par exemple, on a ici le nombre de "backers" dans la colonne
      * "backers_count". Il s'agit du nombre de personnes FINAL ayant investi dans
      * chaque projet, or ce nombre n'est connu qu'après la fin de la campagne. Il faut
      * savoir repérer et traiter ces données pour plusieurs raisons:
      *
      * En pratique quand on voudra appliquer notre modèle, les données du futur ne sont
      * pas présentes (puisqu'elles ne sont pas encore connues). On ne peut donc pas les
      * utiliser comme input pour un modèle.
      *
      * Pendant l'entraînement (si on ne les a pas enlevées) elles facilitent le travail
      * du modèle puisque qu'elles contiennent des informations directement liées à ce
      * qu'on veut prédire. Par exemple, si backers_count = 0 on est sûr que la campagne
      * a raté.
      *
      * Ici, pour enlever les données du futur on retir les colonnes "backers_count" et "state_changed_at".
      */
    println("\nDrop futur data: start")
    val featureDF_dropFutur: DataFrame = featureDF_drop.drop("backers_count", "state_changed_at")
    println("Drop futur data: done.\n")


    /** e)
      *
      */
    println("\nVisualization of countries inside currency column:")
    featureDF_dropFutur.filter($"country".isNull).groupBy("currency").count.orderBy(($"count".desc)).show(nValueToShow)


    // Methode qui renvoit country s'il est non nul, et currency sinon.
    def udf_country = udf{(country: String, currency: String) =>
      if (country == null) // && currency != "false")
        currency
      else
        country //: ((String, String) => String)  pour éventuellement spécifier le type
    }

    //Methode qui filtre les currency: ne renvoie pas la monnaie est non nulle et de taille differente de 3
    def udf_currency = udf{(currency: String) =>
      if ( currency != null && currency.length != 3 )
        null
      else
        currency //: ((String, String) => String)  pour éventuellement spécifier le type
    }

    println("\nApplying udf functions to clean country/currency")
    val dfCountry: DataFrame = featureDF_dropFutur
      .withColumn("country2", udf_country($"country", $"currency"))
      .withColumn("currency2", udf_currency($"currency"))
      .drop("country", "currency")
    println("Udf functions applied.\n")

    dfCountry.groupBy("country2", "currency2").count.orderBy($"count".desc).show(nValueToShow)

    // Pour aider notre algorithme, on souhaite qu'un même mot écrit en minuscules ou majuscules ne soit pas deux
    // "entités" différentes. On met tout en minuscules
    println("\nLowering strings contained in the df")
    val dfLower: DataFrame = dfCountry
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))

    dfLower.show(nValueToShow)

    // Remplacer les strings "false" dans currency et country
    // En observant les colonnes
    dfLower.groupBy("country2").count.orderBy($"count".desc).show(nValueToShow)
    dfLower.groupBy("currency2").count.orderBy($"count".desc).show(nValueToShow)


    /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/

    // a) b) c) features à partir des timestamp
    val dfDurations: DataFrame = dfLower
      .withColumn("deadline2", from_unixtime($"deadline"))
      .withColumn("created_at2", from_unixtime($"created_at"))
      .withColumn("launched_at2", from_unixtime($"launched_at"))
      .withColumn("days_campaign", datediff($"deadline2", $"launched_at2")) // datediff requires a dateType
      .withColumn("hours_prepa", round(($"launched_at" - $"created_at")/3600.0, 3)) // here timestamps are in seconds, there are 3600 seconds in one hour
      .filter($"hours_prepa" >= 0 && $"days_campaign" >= 0)
      .drop("created_at", "deadline", "launched_at")

    // d)
    val dfText= dfDurations.withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))

    /** VALEUR NULLES **/

    val dfReady: DataFrame = dfText
      .filter($"goal" > 0)
      .na
      .fill(Map(
        "days_campaign" -> -1,
        "hours_prepa" -> -1,
        "goal" -> -1
      ))


    // vérifier l'équilibrage pour la classification
    dfReady.groupBy("final_status").count.orderBy($"count".desc).show

    // filtrer les classes qui nous intéressent
    // Final status contient d'autres états que Failed ou Succeed. On ne sait pas ce que sont ces états,
    // on peut les enlever ou les considérer comme Failed également. Seul "null" est ambigue et on les enlève.
    val dfFiltered = dfReady.filter($"final_status".isin(0, 1))

    dfFiltered.show(nValueToShow)
    println(dfFiltered.count)





    /** WRITING DATAFRAME **/

    dfFiltered.write.mode(SaveMode.Overwrite).parquet("/home/bud/Documents/s1/spark2/fundingData/prepared_trainingset")


  /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/


  }

}