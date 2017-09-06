package com.example.config

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SparkSession, SQLContext}

trait SparkHolder {
  def sparkConfig: SparkConf
  def sparkSession: SparkSession
  def sc: SparkContext
  def sqlContext: SQLContext
}

trait LocalSparkHolder extends SparkHolder with ConfigHolder {
  override lazy val sparkConfig = new SparkConf()
  override lazy val sparkSession: SparkSession =
    if (config.getIsNull("spark.master"))
      SparkSession.builder.appName("SparkML").config(sparkConfig).getOrCreate()
    else
      SparkSession.builder.master(config.getString("spark.master"))
        .appName("SparkML").config(sparkConfig).getOrCreate()

  override lazy val sc: SparkContext = sparkSession.sparkContext

  override lazy val sqlContext: SQLContext = sparkSession.sqlContext
}
