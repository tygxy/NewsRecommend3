/**
  * Created by guoxingyu on 2017/6/23.
  */

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD

object NewsRecommend {
  def main(args: Array[String]) {
//    if (args.length < 2) {
//      System.out.println("Usage:<master> <hdfs dir path>")
//      System.exit(1)
//    }

    // 创建入口对象
    /**
      * 测试专用
      */
//    val conf = new SparkConf().setMaster("local[*]").setAppName("Collaborative Filtering")
//    val inputPath = "/Users/guoxingyu/Documents/work/spark/NewsRecommend/input"
//    val outputPath = "/Users/guoxingyu/Documents/work/spark/NewsRecommend/output"
    /**
      * 集群专用
      */
    val conf = new SparkConf().setAppName("Collaborative Filtering")
    val inputPath = "/qcdq/recommend/newsrecommend/baseOnCF/input"
    val outputPath = "/qcdq/recommend/newsrecommend/baseOnCF/output"

    val sc = new SparkContext(conf)

    // 读取评分数据为RDD
    val ratings: RDD[(Rating, String)] = sc.textFile(inputPath).map { lines =>
      val fields = lines.split(",")
      val rating = Rating(fields(0).toInt, fields(2).toInt, fields(3).toDouble)
      val username = fields(1)
      (rating,username)
    }

    val allRatings: RDD[(Rating, String)] = ratings.cache()


    // 获取用户信息
    val userDict: Map[Int, String] = sc.textFile(inputPath).map { lines =>
      val fields = lines.split(",")
      (fields(0).toInt, fields(1))
    }.collect().toMap

    // 获取新闻信息
    // val newsSet = sc.textFile(args(0)).map { lines =>
    //   val fields = lines.split(",")
    //   (fields(2).toInt)
    // }.collect().toSeq

    // 输出统计信息
    // val numRatings = ratings.count()
    // val numUsers = ratings.map(_._1.user).distinct().count()
    // val numMovies = ratings.map(_._1.product).distinct().count()
    // print("get " + numRatings+" ratings from " + numUsers + " users on " + numMovies + " movies ")


    // 定义函数计算均方误差RMSE
    //    def computeRmse(model : MatrixFactorizationModel, data: RDD[Rating]) :
    //    Double = {
    //      val predictions : RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    //      val predictionsAndRatings  = predictions.map{x =>
    //        ((x.user, x.product), x.rating)
    //      }.join(data.map(x => ((x.user, x.product), x.rating))).values
    //      math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
    //    }
    //    val newmode = ALS.train(training, 8, 20, 10.0)


    // 训练模型参数
    //    val ranks = List(8,9)
    //    val lambdas = List(9.0, 10,0)
    //    val numIters = List(19, 20)
    //    var bestModel : Option[MatrixFactorizationModel] = None
    //    var bestValdationRmse = Double.MaxValue
    //    var bestRank = 0
    //    var bestLambda = -1.0
    //    var bestNumIter = -1
    //    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
    //      val model = ALS.train(ratings, rank, numIter, lambda)
    //      val valalidationRmse = computeRmse(model, validation)
    //      if (valalidationRmse < bestValdationRmse) {
    //        bestModel = Some(model)
    //        bestValdationRmse = valalidationRmse
    //        bestRank = rank
    //        bestLambda = lambda
    //        bestNumIter = numIter
    //      }
    //    }
    //    val testRmse = computeRmse(bestModel.get , test)
    //    println("The best model was trained with rank " + bestRank + " and lambda " +
    //            bestLambda + " and numIter = " + bestNumIter + " , and its RMSE on the test set is " +
    //            testRmse + ".")

    // 获取模型
    // val bestmodel = ALS.train(ratings.map(_._1), 8, 15, 0.01)
    val bestmodel = ALS.trainImplicit(allRatings.map(_._1), 20, 10, 0.01,1.0)


    // 为每个用户推荐10部电影
    val allRecommendations = bestmodel.recommendProductsForUsers(10).map{
      case (userId,recommends) =>
        val str = new StringBuilder()
        for (r <- recommends) {
          if (str.nonEmpty) {
            str.append(" ")
          }
          var tmp = r.rating.toString.substring(0,4).toDouble
          str.append(r.product).append("=").append(tmp)
        }
         userDict.get(userId).get + "," +str.toString()
    }

    // 查看结果
    // allRecommendations.take(100).foreach(r =>
    //   println(r)
    // )

    // 输出结果
    allRecommendations.saveAsTextFile(outputPath)


  }
}
