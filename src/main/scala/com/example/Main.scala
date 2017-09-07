package com.example

import com.example.config.{DefaultConfigHolder, LocalSparkHolder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataTypes, DoubleType, IntegerType}

import scala.util.matching.Regex

object Main extends LocalSparkHolder with DefaultConfigHolder {

	val objectsLocation = "/home/ilia/Documents/bidata_docs/task10/session.dataset/Objects.csv"
	val choicesLocation = "/home/ilia/Documents/bidata_docs/task10/session.dataset/Target.csv"
	val descLocation = "/home/ilia/Documents/bidata_docs/task10/session.dataset/PropertyDesciptionEN.txt"

	def main(args: Array[String]): Unit = {

		val sparkContext = sparkSession.sparkContext
		val sqLContext = sparkSession.sqlContext
		val ssc = new StreamingContext(sparkContext, Seconds(10))

		val filterRegex = "[0-9]\\)".r

		import sqLContext.implicits._

		val descrRdd = sparkContext
			.textFile(descLocation)
			.filter(str => str.trim.nonEmpty)
			.filter(str => filterRegex.findFirstIn(str).isDefined)
			.map(str => splitByRegex(str, ".*\\)\\s(.*):.*".r))
			.collect().toSeq.dropRight(1)

		val choicesDF = sparkContext
			.textFile(choicesLocation)
			.zipWithIndex()
			.map { case (line, id) => (id, line.toInt) }
			.toDF("Id", "label")

		val objects = sqLContext
			.read
			.format("com.databricks.spark.csv")
			.option("header", "false")
			.option("inferSchema", "true")
			.option("delimiter", ";")
			.load(objectsLocation)
			.toDF(descrRdd: _*)
			.withColumn("Id", monotonically_increasing_id())

		val objectsWithDeceison = objects
			.join(choicesDF, "Id")
			.drop("Id")
			.withColumn("Personal income", objects("Personal income").cast(DataTypes.IntegerType))
			.withColumn("The amount of the last loan", objects("The amount of the last loan").cast(IntegerType))
			.withColumn("Down payment", objects("Down payment").cast(IntegerType))
			.withColumn("Average amount of the delayed payment, USD", objects("Average amount of the delayed payment, USD").cast(DoubleType))
			.withColumn("Maximum amount of the delayed payment, USD", objects("Maximum amount of the delayed payment, USD").cast(DoubleType))
			.drop(col("The number of utilized cards"))
			.filter(row => !row.anyNull)

		val labelIndexer = new StringIndexer()
			.setInputCol("label")
			.setOutputCol("indexedLabel")
			.fit(objectsWithDeceison)

		val vectorAssembler = new VectorAssembler()
			.setInputCols(descrRdd.dropRight(1).toArray)
			.setOutputCol("features")

		objectsWithDeceison.limit(20).foreach(obj => println(obj))

		val transformedObjects = vectorAssembler.transform(objectsWithDeceison)

		val featureIndexer = new VectorIndexer()
			.setInputCol("features")
			.setOutputCol("indexedFeatures")
			.setMaxCategories(10)
			.fit(transformedObjects)

		val Array(trainingData, testData) = transformedObjects.randomSplit(Array(0.7, 0.3))

		val rf = new RandomForestClassifier()
			.setLabelCol("indexedLabel")
			.setFeaturesCol("indexedFeatures")
			.setNumTrees(10)

		val labelConverter = new IndexToString()
			.setInputCol("prediction")
			.setOutputCol("predictedLabel")
			.setLabels(labelIndexer.labels)

		val pipeline = new Pipeline()
			.setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

		val model = pipeline.fit(trainingData)

		val predictions = model.transform(testData)

		predictions.select("predictedLabel", "label", "features").show(5)

		val evaluator = new MulticlassClassificationEvaluator()
			.setLabelCol("indexedLabel")
			.setPredictionCol("prediction")
			.setMetricName("accuracy")
		val accuracy = evaluator.evaluate(predictions)
		println("Test Error = " + (1.0 - accuracy))

		val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
		println("Learned classification forest model:\n" + rfModel.toDebugString)
	}

	def splitByRegex(str: String, r: Regex) = str match {
		case r(group) => group
	}
}
