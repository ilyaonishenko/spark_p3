package com.example



import com.example.config.{DefaultConfigHolder, LocalSparkHolder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DataTypes, DoubleType, IntegerType}
import org.apache.spark.sql.functions._

import scala.util.matching.Regex

object Main extends LocalSparkHolder with DefaultConfigHolder {

	val objectsLocation = "/home/ilia/Documents/bidata_docs/task10/session.dataset/Objects.csv"
	val choicesLocation = "/home/ilia/Documents/bidata_docs/task10/session.dataset/Target.csv"
	val descLocation = "/home/ilia/Documents/bidata_docs/task10/session.dataset/PropertyDesciptionEN.txt"

	def main(args: Array[String]): Unit = {

		val sparkContext = sparkSession.sparkContext
		val sqLContext = sparkSession.sqlContext

		val filterRegex = "[0-9]\\)".r

		import sqLContext.implicits._

		val descrRdd = sparkContext
			.textFile(descLocation)
			.filter(str => str.trim.nonEmpty)
			.filter(str => filterRegex.findFirstIn(str).isDefined)
			.map(str => splitByRegex(str, ".*\\)\\s(.*):.*".r))
			.collect()
			.toSeq
			.dropRight(1)

		val choicesDF = sparkContext
			.textFile(choicesLocation)
			.zipWithIndex()
			.map { case (line, id) => (id, line.toDouble) }
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
			.na.drop()

		val vectorAssembler = new VectorAssembler()
			.setInputCols(descrRdd.dropRight(1).toArray)
			.setOutputCol("features")

		val transformedObjects = vectorAssembler
			.transform(objectsWithDeceison)
	  	.drop(descrRdd.dropRight(1).toArray:_*)

		transformedObjects.printSchema()

//		transformedObjects.select(col("features")).limit(2).foreach(row => println(row.getAs[Vector](0).size))

		val Array(trainingData, testData) = transformedObjects.randomSplit(Array(0.75, 0.25))

		// Index labels, adding metadata to the label column.
		// Fit on whole dataset to include all labels in index.
//		val labelIndexer = new StringIndexer()
//			.setInputCol("label")
//			.setOutputCol("indexedLabel")
//			.fit(transformedObjects)
//		// Automatically identify categorical features, and index them.
//		val featureIndexer = new VectorIndexer()
//			.setInputCol("features")
//			.setOutputCol("indexedFeatures")
//			.setMaxCategories(22) // features with > 4 distinct values are treated as continuous.
//			.fit(transformedObjects)


		// Train a DecisionTree model.
//		val dt = new DecisionTreeClassifier()
//			.setLabelCol("indexedLabel")
//			.setFeaturesCol("indexedFeatures")
//
//		// Convert indexed labels back to original labels.
//		val labelConverter = new IndexToString()
//			.setInputCol("prediction")
//			.setOutputCol("predictedLabel")
//			.setLabels(labelIndexer.labels)
//
//		// Chain indexers and tree in a Pipeline.
//		val pipeline = new Pipeline()
//			.setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
//
//		// Train model. This also runs the indexers.
//		val model = pipeline.fit(trainingData)
//
//		// Make predictions.
//		val predictions = model.transform(testData)
//
//		// Select example rows to display.

		val numFeatures = 49

		val lr = new LogisticRegression()
			.setMaxIter(500)
//			.setRegParam(0.3)
			.setElasticNetParam(0.3)
			.fit(trainingData)

		println(s"Coefficients: ${lr.coefficients} Intercept: ${lr.intercept}")

		val trainingSummary = lr.summary
		val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
		val roc = binarySummary.roc
		val auc = binarySummary.areaUnderROC

		println("roc: " + roc)
		println("AUC: " + auc)

		val fMeasure: DataFrame = binarySummary.fMeasureByThreshold
		val maxFMeasure: Double = fMeasure.select(max("F-Measure")).head().getDouble(0)
		val bestThreshold: Double = fMeasure.where($"F-Measure" === maxFMeasure)
			.select("threshold").head().getDouble(0)
		lr.setThreshold(bestThreshold)

		val summary = lr.evaluate(testData)

		println("probability: " + summary)
		println(" predictions: ")
		summary
			.predictions.select(col("label"), col("prediction"))
			.write
	  	.format("com.databricks.spark.csv")
			.option("header", "true")
			.save("/home/ilia/Documents/bidata_docs/task10/session.dataset/res")
//		predictions.printSchema()
//	  	.select(col("probability"))
//		predictions.printSchema()
//		val n = predictions.first.getAs[org.apache.spark.ml.linalg.Vector](0).size
//		val vecToSeq = udf((v: Vector) => v.toArray)
//		val exprs = (0 until n).map(i => $"_tmp".getItem(i).alias(s"f$i"))
//
//		predictions
//			.select(vecToSeq($"probability").alias("_tmp"))
//			.select(exprs:_*)
//			.write
//			.format("com.databricks.spark.csv")
//			.option("header", "true")
//			.save("/home/ilia/Documents/bidata_docs/task10/session.dataset/res")
	}

	def splitByRegex(str: String, r: Regex): String = str match {
		case r(group) => group
	}
}
