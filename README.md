# PlayerDecision

&emsp;&emsp;A concept example shows a basic method to generate game AI based on human input with game play data.
&emsp;&emsp;The example training data are 100 records of random generated features (represent different game situations) with manually determined label data (Player_Decision).
&emsp;&emsp;With decision tree classification algorithm, the outputs are learned decision tree model which tells all kinds of actions the player will take for all the specified situations, model training accuracy and important data features that affect the decisions.
&emsp;&emsp;The “GameSettings.csv” file just help with manually labeling and is not for training use. The training algorithm is done by Apache Spark framework with Scala.


