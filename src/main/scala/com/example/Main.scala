package com.example

import com.example.config.{DefaultConfigHolder, LocalSparkHolder}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row

object Main extends App with LocalSparkHolder with DefaultConfigHolder  {
	val training = sparkSession.createDataFrame(Seq(
		(1.0, Vectors.dense(0.0, 1.1, 0.1)),
		(0.0, Vectors.dense(2.0, 1.0, -1.0)),
		(0.0, Vectors.dense(2.0, 1.3, 1.0)),
		(1.0, Vectors.dense(0.0, 1.2, -0.5))
	)).toDF("label", "features")

	// Create a LogisticRegression instance. This instance is an Estimator.
	val lr = new LogisticRegression()
	// Print out the parameters, documentation, and any default values.
	println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

	// We may set parameters using setter methods.
	lr.setMaxIter(10)
		.setRegParam(0.01)

	// Learn a LogisticRegression model. This uses the parameters stored in lr.
	val model1 = lr.fit(training)
	// Since model1 is a Model (i.e., a Transformer produced by an Estimator),
	// we can view the parameters it used during fit().
	// This prints the parameter (name: value) pairs, where names are unique IDs for this
	// LogisticRegression instance.
	println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)

	// We may alternatively specify parameters using a ParamMap,
	// which supports several methods for specifying parameters.
	val paramMap = ParamMap(lr.maxIter -> 20)
		.put(lr.maxIter, 30)  // Specify 1 Param. This overwrites the original maxIter.
		.put(lr.regParam -> 0.1, lr.threshold -> 0.55)  // Specify multiple Params.

	// One can also combine ParamMaps.
	val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")  // Change output column name.
	val paramMapCombined = paramMap ++ paramMap2

	// Now learn a new model using the paramMapCombined parameters.
	// paramMapCombined overrides all parameters set earlier via lr.set* methods.
	val model2 = lr.fit(training, paramMapCombined)
	println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)

	// Prepare test data.
	val test = sparkSession.createDataFrame(Seq(
		(1.0, Vectors.dense(-1.0, 1.5, 1.3)),
		(0.0, Vectors.dense(3.0, 2.0, -0.1)),
		(1.0, Vectors.dense(0.0, 2.2, -1.5))
	)).toDF("label", "features")

	// Make predictions on test data using the Transformer.transform() method.
	// LogisticRegression.transform will only use the 'features' column.
	// Note that model2.transform() outputs a 'myProbability' column instead of the usual
	// 'probability' column since we renamed the lr.probabilityCol parameter previously.
	model2.transform(test)
		.select("features", "label", "myProbability", "prediction")
		.collect()
		.foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
			println(s"($features, $label) -> prob=$prob, prediction=$prediction")
		}
	/*object Config {
		@Parameter(names = Array("-st", "--slackToken"))
		var slackToken: String = null
		@Parameter(names = Array("-nc", "--numClusters"))
		var numClusters: Int = 4
		@Parameter(names = Array("-po", "--predictOutput"))
		var predictOutput: String = null
		@Parameter(names = Array("-td", "--trainData"))
		var trainData: String = null
		@Parameter(names = Array("-ml", "--modelLocation"))
		var modelLocation: String = null
	}*/

//	def main(args: Array[String]): Unit ={

//		val sparkSession = SparkSession
//			.builder()
//			.appName("sparkML")
//			.enableHiveSupport()
//			.getOrCreate()
//
//		val sparkContext = sparkSession.sparkContext
//		val sqLContext = sparkSession.sqlContext
//		val ssc = new StreamingContext(sparkContext, Seconds(10))
//
//		import sqLContext.implicits._
//		val descrRdd = sparkContext.textFile(args(1))
//		val objectsDF: DataFrame = sparkContext.textFile(args(0)).toDF()
//		val choicesDF: DataFrame = sparkContext.textFile(args(2)).toDF()








//		ssc.start()
//	}

//	def mms(args: Array[String]) {
//		val conf = new SparkConf().setAppName("SlackStreamingWithML")
//		val sparkContext = new SparkContext(conf)
//
//		// optain existing or create new model
//		val clusters: KMeansModel =
//			if (Config.trainData != null) {
//				KMeanTrainTask.train(sparkContext, Config.trainData, Config.numClusters, Config.modelLocation)
//			} else {
//				if (Config.modelLocation != null) {
//					new KMeansModel(sparkContext.objectFile[Vector](Config.modelLocation).collect())
//				} else {
//					throw new IllegalArgumentException("Either modelLocation or trainData should be specified")
//				}
//			}
//
//		if (Config.slackToken != null) {
//			SlackStreamingTask.run(sparkContext, Config.slackToken, clusters, Config.predictOutput)
//		}
//
//	}
//
//	def train(sparkContext: SparkContext, trainData: String, numClusters: Int, modelLocation: String): KMeansModel = {
//
//		if (new File(modelLocation).exists) removePrevious(modelLocation)
//
//		val trainRdd = sparkContext.textFile(trainData)
//
//		val parsedData = trainRdd.map(Utils.featurize).cache()
//		// if we had a really large data set to train on, we'd want to call an action to trigger cache.
//
//		val model = KMeans.train(parsedData, numClusters, numIterations)
//
//		sparkContext.makeRDD(model.clusterCenters, numClusters).saveAsObjectFile(modelLocation)
//
//		val example = trainRdd.sample(withReplacement = false, 0.1).map(s => (s, model.predict(Utils.featurize(s)))).collect()
//		println("Prediction examples:")
//		example.foreach(println)
//
//		model
//	}

}
//object Utils {
//
//	val NUM_DEMENSIONS: Int = 1000
//
//	val tf = new HashingTF(NUM_DEMENSIONS)
//
//	/**
//		* This uses min hash algorithm https://en.wikipedia.org/wiki/MinHash to transform
//		* string to vector of double, which is required for k-means
//		*/
//	def featurize(s: String): Vector = {
//		tf.transform(s.sliding(2).toSeq)
//	}
//
//}
//
//object SlackStreamingTask {
//
//	def run(sparkContext: SparkContext, slackToken: String, clusters: KMeansModel, predictOutput: String) {
//		val ssc = new StreamingContext(sparkContext, Seconds(5))
//		val dStream = ssc.receiverStream(new SlackReceiver(slackToken))
//
//		val stream = dStream //create stream of events from the Slack... but filter and marshall to JSON stream data
//			.filter(JSON.parseFull(_).get.asInstanceOf[Map[String, String]]("type") == "message") // get only message events
//			.map(JSON.parseFull(_).get.asInstanceOf[Map[String, String]]("text")) // extract message text from the event
//
//		val kmeanStream = kMean(stream, clusters) // create K-mean model
//		kmeanStream.print() // print k-mean results. It is pairs (k, m), where k - is a message text, m - is a cluster number to which message relates
//
//		if (predictOutput != null) {
//			kmeanStream.saveAsTextFiles(predictOutput) // save to results to the file, if file name specified
//		}
//
//		ssc.start() // run spark streaming application
//		ssc.awaitTermination() // wait the end of the application
//	}
//
//	/**
//		* transform stream of strings to stream of (string, vector) pairs and set this stream as input data for prediction
//		*/
//	def kMean(dStream: DStream[String], clusters: KMeansModel): DStream[(String, Int)] = {
//		dStream.map(s => (s, Utils.featurize(s))).map(p => (p._1, clusters.predict(p._2)))
//	}
//
//}

