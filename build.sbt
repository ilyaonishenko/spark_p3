name := "spark_p3"

version := "0.1"

scalaVersion := "2.11.8"

resolvers += "Clojars" at "http://clojars.org/repo"

assemblyMergeStrategy in assembly := {
	case PathList("META-INF", xs @ _*) => MergeStrategy.discard
	case x => MergeStrategy.first
}

libraryDependencies ++= Seq(
	"com.typesafe" % "config" % "1.3.1",
	"org.apache.hadoop" % "hadoop-common" % "2.7.3" % "compile",
	"org.apache.spark" % "spark-core_2.11" % "2.2.0" % "compile",
	"org.apache.spark" % "spark-streaming_2.11" % "2.2.0" % "compile",
	"org.apache.spark" % "spark-sql_2.11" % "2.2.0" % "compile",
	"org.apache.spark" % "spark-mllib_2.11" % "2.2.0" % "compile"
)