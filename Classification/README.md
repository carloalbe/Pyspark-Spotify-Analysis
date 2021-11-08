```python
!pip install pyspark==3.1.2 py4j==0.10.9
```

    Collecting pyspark==3.1.2
      Downloading pyspark-3.1.2.tar.gz (212.4 MB)
    [K     |████████████████████████████████| 212.4 MB 64 kB/s 
    [?25hCollecting py4j==0.10.9
      Downloading py4j-0.10.9-py2.py3-none-any.whl (198 kB)
    [K     |████████████████████████████████| 198 kB 54.8 MB/s 
    [?25hBuilding wheels for collected packages: pyspark
      Building wheel for pyspark (setup.py) ... [?25l[?25hdone
      Created wheel for pyspark: filename=pyspark-3.1.2-py2.py3-none-any.whl size=212880768 sha256=b66cc046da3d8dcac4b078170f3275907c99b8d501b84d48b650249645120063
      Stored in directory: /root/.cache/pip/wheels/a5/0a/c1/9561f6fecb759579a7d863dcd846daaa95f598744e71b02c77
    Successfully built pyspark
    Installing collected packages: py4j, pyspark
    Successfully installed py4j-0.10.9 pyspark-3.1.2
    


```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.stat import Correlation
from pyspark.sql.types import IntegerType,BooleanType,DateType,FloatType

spark = SparkSession.builder\
        .master("local[*]")\
        .appName('spotify_classification')\
        .getOrCreate()
```


```python
from google.colab import drive               
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
df = spark.read.json("drive/My Drive/dataset.json")
```


```python
df.show(5)
```

    +------------+------------------+--------------------+---------------------+------------+-----------+------+--------------------+--------------------+----------------+---+--------+--------+----+----------------+------------+-----------+--------------------+---------------------+-------+--------------+-------+
    |acousticness|               age|avg_artist_followers|avg_artist_popularity|danceability|duration_ms|energy|              genres|            id_track|instrumentalness|key|liveness|loudness|mode|popularity_track|release_date|speechiness|sum_artist_followers|sum_artist_popularity|  tempo|time_signature|valence|
    +------------+------------------+--------------------+---------------------+------------+-----------+------+--------------------+--------------------+----------------+---+--------+--------+----+----------------+------------+-----------+--------------------+---------------------+-------+--------------+-------+
    |       0.658| 41.83013698630137|              5403.5|                 40.0|       0.602|     156067| 0.552|[classic czech po...|00AeAaSNbe92PRrst...|             0.0|  0|  0.0972|  -6.667|   1|               3|  1980-01-01|      0.404|               10807|                   80|182.229|             3|   0.65|
    |       0.543| 45.83287671232877|             19833.0|                 43.0|        0.77|     220133| 0.891|[afrobeat, afropo...|00DJt4PjkzeXhKKVD...|         7.96E-4|  1|  0.0684|  -7.306|   1|               9|  1976-01-01|      0.172|               19833|                   43|135.573|             4|  0.898|
    |      4.8E-5|25.682191780821917|            874600.0|                 68.0|       0.212|     250960| 0.986|[alternative meta...|00HgVIkZrAL8WjAN9...|           0.918|  0|   0.324|   -6.69|   0|              33|  1996-02-20|       0.14|              874600|                   68|140.917|             4|  0.231|
    |       0.144| 31.82191780821918|             69129.0|                 42.0|       0.362|     457040| 0.453|[corrosion, dark ...|00Lx2f8NRiFKMGWa0...|           0.827| 11|   0.117| -17.744|   0|              35|  1990-01-01|     0.0398|               69129|                   42|118.853|             4|  0.257|
    |       0.957| 4.079452054794521|           1709414.0|                 68.0|       0.343|     282891| 0.225|[brazilian rock, ...|00fzPML4lrNag28OP...|         2.49E-4|  1|   0.661| -14.937|   0|              52|  2017-09-22|     0.0384|             1709414|                   68|144.533|             4|  0.101|
    +------------+------------------+--------------------+---------------------+------------+-----------+------+--------------------+--------------------+----------------+---+--------+--------+----+----------------+------------+-----------+--------------------+---------------------+-------+--------------+-------+
    only showing top 5 rows
    
    


```python
from pyspark.ml.feature import QuantileDiscretizer 

qds = QuantileDiscretizer(relativeError=0.0001, handleInvalid="error", numBuckets=10, inputCol="popularity_track", outputCol="label")

df = qds.setHandleInvalid("keep").fit(df).transform(df)
```


```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler( 
inputCols=[
 'age',
 'duration_ms',
 'danceability',
 'energy',
 'key',
 'mode',
 'speechiness',
 'acousticness',
 'instrumentalness',
 'liveness',
 'valence',
 'tempo',
 'time_signature',
 'sum_artist_followers',
 'sum_artist_popularity',
 'avg_artist_popularity'], 
outputCol="feat")
df=assembler.transform(df)
```


```python
from pyspark.ml.feature import StandardScaler


scaler = StandardScaler(inputCol="feat", outputCol="features")


scalerModel = scaler.fit(df)


df = scalerModel.transform(df)
```


```python
final_data = df.select("id_track", "features","label")
```


```python
train, test = final_data.randomSplit([0.7, 0.3], seed = 10)
```


```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
```

# Random Forest


```python
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol="features",labelCol="label",maxDepth = 10)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
```


```python
rfModel.featureImportances
```




    SparseVector(16, {0: 0.4257, 1: 0.0338, 2: 0.0099, 3: 0.0208, 4: 0.0013, 5: 0.0003, 6: 0.0235, 7: 0.0684, 8: 0.0225, 9: 0.0057, 10: 0.0066, 11: 0.0029, 12: 0.0008, 13: 0.1089, 14: 0.1148, 15: 0.1541})




```python
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
evaluator.evaluate(predictions)
```




    0.35512458112020423




```python
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="f1")
evaluator.evaluate(predictions)
```




    0.329023689904405




```python
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="recallByLabel") 
evaluator.evaluate(predictions)
```




    0.7929934424292551




```python
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="precisionByLabel") 
evaluator.evaluate(predictions)
```




    0.7389892094252367



# MLP


```python
from pyspark.ml.classification import MultilayerPerceptronClassifier
```


```python
layers = [16,10,3,10]
```


```python
mlp = MultilayerPerceptronClassifier(labelCol='label',
                                            featuresCol='features',
                                            maxIter=100,
                                            layers=layers,
                                            seed=1234)

mlpModel = mlp.fit(train)


predictions = mlpModel.transform(test)
```


```python
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
evaluator.evaluate(predictions)
```




    0.2928454191077575




```python
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="f1")
evaluator.evaluate(predictions)
```




    0.25047767662548587




```python
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="recallByLabel") 
evaluator.evaluate(predictions)
```




    0.6647958882259113




```python
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="precisionByLabel") 
evaluator.evaluate(predictions)
```




    0.6857404021937843




```python

```
