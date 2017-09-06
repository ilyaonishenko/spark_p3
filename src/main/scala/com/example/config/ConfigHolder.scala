package com.example.config

import java.util.Properties

import com.typesafe.config.{Config, ConfigFactory}

trait ConfigHolder {
  def config: Config

  def propsFromConfig(config: Config): Properties = {
    import scala.collection.JavaConversions._

    val props = new Properties()

    val map: Map[String, Object] = config.entrySet().map({ entry =>
      entry.getKey -> entry.getValue.unwrapped()
    })(collection.breakOut)

    props.putAll(map)
    props
  }
}

trait DefaultConfigHolder {
  val config: Config = ConfigFactory.defaultApplication()
}
