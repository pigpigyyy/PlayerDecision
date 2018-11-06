import org.apache.spark.ml.{Pipeline,PipelineStage}
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,StringIndexerModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier,DecisionTreeClassificationModel}
import org.apache.spark.ml.tree._
import org.apache.spark.mllib.tree.configuration.FeatureType._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val df = spark.read.options(Map("inferSchema"->"true","header"->"true")).csv("PlayerData.csv")

val actionColumns = Array("Nearest_Enemy_Action","Current_Player_Action","Player_Decision")

val stringIndexers = actionColumns.map(name => new StringIndexer().setInputCol(name).setOutputCol(s"${name}_Index").fit(df).asInstanceOf[PipelineStage])
val continuousNames = df.columns.filter(c => actionColumns.find(ac => ac.equals(c)) == None)
val featureNames = continuousNames.union(actionColumns.filter(!_.equals("Player_Decision")).map(c => s"${c}_Index"))

val assembler = new VectorAssembler()
    .setInputCols(featureNames)
    .setOutputCol("features")

val pipeline = new Pipeline()
    .setStages(stringIndexers :+ assembler.asInstanceOf[PipelineStage])

val inputDf = pipeline.fit(df).transform(df)

val decisionTreeTrainer = new DecisionTreeClassifier()
    .setMaxDepth(5)
    .setMaxBins(32)
    .setImpurity("gini")
    .setMinInstancesPerNode(1)
    .setFeaturesCol("features")
    .setLabelCol("Player_Decision_Index")

val model = decisionTreeTrainer.fit(inputDf)

val outputFeatures = continuousNames.union(actionColumns.filter(!_.equals("Player_Decision")))
val indexerMap = (for (i <- 0 to actionColumns.length-1) yield (actionColumns(i), stringIndexers(i).asInstanceOf[StringIndexerModel])).toMap
val conv = outputFeatures.map(f => if (actionColumns.find(ac => ac.equals(f)) == None) Array[String]() else indexerMap(f).labels)
val predictionConv = indexerMap("Player_Decision").labels

def treeToString(node: Node, indent: Int = 0): String = {
    def splitToString(split: Split, left: Boolean): String = {
        split match {
            case contSplit: ContinuousSplit => if (left) {
                s"(${outputFeatures(contSplit.featureIndex)} <= ${contSplit.threshold})"
            } else {
                s"(${outputFeatures(contSplit.featureIndex)} > ${contSplit.threshold})"
            }
            case catSplit: CategoricalSplit => if (left) {
                s"(${outputFeatures(catSplit.featureIndex)} is ${catSplit.leftCategories.toList.map(x => conv(catSplit.featureIndex)(x.toInt)).mkString("{", ",", "}")})"
            } else {
                s"(${outputFeatures(catSplit.featureIndex)} is ${catSplit.rightCategories.toList.map(x => conv(catSplit.featureIndex)(x.toInt)).mkString("{", ",", "}")})"
            }
        }
    }
    val prefix: String = "\t" * indent
    node match {
        case n: LeafNode => prefix + s"Prediction: ${predictionConv(n.prediction.toInt)}\n"
        case n: InternalNode => prefix + s"If ${splitToString(n.split, left = true)}\n" +
            treeToString(n.leftChild, indent + 1) +
            prefix + s"Else ${splitToString(n.split, left = false)}\n" +
            treeToString(n.rightChild, indent + 1)
    }
}

println(treeToString(model.rootNode))

val predictions = model.transform(inputDf)

val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("Player_Decision_Index")

println("accuracy："+evaluator.setMetricName("accuracy").evaluate(predictions))
println("f1："+evaluator.setMetricName("f1").evaluate(predictions))

val fs = model.featureImportances.toArray
val cols = featureNames
val featureImportances = (for (i <- 0 to cols.length-1) yield (i, fs(i))).toList.sortWith(_._2 > _._2)
println("Feature Importances：")
featureImportances.foreach(x => println(s"${cols(x._1)}：${x._2}"))

