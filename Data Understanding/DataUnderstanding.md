# Data Understanding

In this section we'll present how we've studied, cleaned, integrated and explored our datasets.

## Importing Libraries 




```python
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns

import pandas as pd
import numpy as np
```

#### Pyspark tools


```python
import pyspark

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.ml.stat import Correlation
from pyspark.sql.types import IntegerType,BooleanType,DateType,FloatType, StringType
from  pyspark.sql.functions import *
```

### Starting Pyspark Session

The configuation used, allow to use 4 cores in the local machine, and set a memory limit of 8GB of ram.


```python
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .master("local[4]") \
    .config("spark.driver.maxResultSize", "8g") \
    .getOrCreate()

spark
```





    <div>
        <p><b>SparkSession - in-memory</b></p>

<div>
    <p><b>SparkContext</b></p>

    <p><a href="http://DESKTOP-FIMTLSS.station:4040">Spark UI</a></p>

    <dl>
      <dt>Version</dt>
        <dd><code>v2.4.0</code></dd>
      <dt>Master</dt>
        <dd><code>local[4]</code></dd>
      <dt>AppName</dt>
        <dd><code>Python Spark SQL basic example</code></dd>
    </dl>
</div>

    </div>





## Data Loading

### Tracks



```python
artist_df = spark.read.option("header", "true").csv("../data/artists.csv")
print(artist_df.count())
artist_df.printSchema()
artist_df.show(10)
```

    1104349
    root
     |-- id: string (nullable = true)
     |-- followers: string (nullable = true)
     |-- genres: string (nullable = true)
     |-- name: string (nullable = true)
     |-- popularity: string (nullable = true)
    
    +--------------------+---------+------+--------------------+----------+
    |                  id|followers|genres|                name|popularity|
    +--------------------+---------+------+--------------------+----------+
    |0DheY5irMjBUeLybb...|      0.0|    []|Armid & Amir Zare...|         0|
    |0DlhY15l3wsrnlfGi...|      5.0|    []|         ปูนา ภาวิณี|         0|
    |0DmRESX2JknGPQyO1...|      0.0|    []|               Sadaa|         0|
    |0DmhnbHjm1qw6NCYP...|      0.0|    []|           Tra'gruda|         0|
    |0Dn11fWM7vHQ3rinv...|      2.0|    []|Ioannis Panoutsop...|         0|
    |0DotfDlYMGqkbzfBh...|      7.0|    []|       Astral Affect|         0|
    |0DqP3bOCiC48L8SM9...|      1.0|    []|           Yung Seed|         0|
    |0Drs3maQb99iRglyT...|      0.0|    []|               Wi'Ma|         0|
    |0DsPeAi1gxPPnYjgp...|      0.0|    []|             lentboy|         0|
    |0DtvnTxgZ9K5YaPS5...|     20.0|    []|            addworks|         0|
    +--------------------+---------+------+--------------------+----------+
    only showing top 10 rows
    
    


```python
tracks_df = spark.read.option("header", "true").csv("../data/tracks.csv")
tracks_df.show()
tracks_df.printSchema()
tracks_df.count()
```

    +--------------------+--------------------+----------+-----------+--------+-------------------+--------------------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-------+--------------+
    |                  id|                name|popularity|duration_ms|explicit|            artists|          id_artists|release_date|danceability|energy|key|loudness|mode|speechiness|acousticness|instrumentalness|liveness|valence|  tempo|time_signature|
    +--------------------+--------------------+----------+-----------+--------+-------------------+--------------------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-------+--------------+
    |35iwgR4jXetI318WE...|               Carve|         6|     126903|       0|            ['Uli']|['45tIt06XoI0Iio4...|  1922-02-22|       0.645| 0.445|  0| -13.338|   1|      0.451|       0.674|           0.744|   0.151|  0.127|104.851|             3|
    |021ht4sdgPcrDgSk7...|Capítulo 2.16 - B...|         0|      98200|       0|['Fernando Pessoa']|['14jtPCOoNZwquk5...|  1922-06-01|       0.695| 0.263|  0| -22.136|   1|      0.957|       0.797|             0.0|   0.148|  0.655|102.009|             1|
    |07A5yehtSnoedViJA...|Vivo para Querert...|         0|     181640|       0|['Ignacio Corsini']|['5LiOoJbxVSAMkBS...|  1922-03-21|       0.434| 0.177|  1|  -21.18|   1|     0.0512|       0.994|          0.0218|   0.212|  0.457|130.418|             5|
    |08FmqUhxtyLTn6pAh...|El Prisionero - R...|         0|     176907|       0|['Ignacio Corsini']|['5LiOoJbxVSAMkBS...|  1922-03-21|       0.321|0.0946|  7| -27.961|   1|     0.0504|       0.995|           0.918|   0.104|  0.397| 169.98|             3|
    |08y9GfoqCWfOGsKdw...| Lady of the Evening|         0|     163080|       0|    ['Dick Haymes']|['3BiJGZsyX9sJchT...|        1922|       0.402| 0.158|  3|   -16.9|   0|      0.039|       0.989|            0.13|   0.311|  0.196| 103.22|             4|
    |0BRXJHRNGQ3W4v9fr...|           Ave Maria|         0|     178933|       0|    ['Dick Haymes']|['3BiJGZsyX9sJchT...|        1922|       0.227| 0.261|  5| -12.343|   1|     0.0382|       0.994|           0.247|  0.0977| 0.0539|118.891|             4|
    |0Dd9ImXtAtGwsmsAD...|      La Butte Rouge|         0|     134467|       0|  ['Francis Marty']|['2nuMRGzeJ5jJEKl...|        1922|        0.51| 0.355|  4| -12.833|   1|      0.124|       0.965|             0.0|   0.155|  0.727| 85.754|             5|
    |0IA0Hju8CAgYfV1hw...|             La Java|         0|     161427|       0|    ['Mistinguett']|['4AxgXfD7ISvJSTO...|        1922|       0.563| 0.184|  4| -13.757|   1|     0.0512|       0.993|        1.55e-05|   0.325|  0.654|133.088|             3|
    |0IgI1UCz84pYeVetn...|  Old Fashioned Girl|         0|     310073|       0|    ['Greg Fieler']|['5nWlsH5RDgFuRAi...|        1922|       0.488| 0.475|  0| -16.222|   0|     0.0399|        0.62|         0.00645|   0.107|  0.544|139.952|             4|
    |0JV4iqw2lSKJaHBQZ...|Martín Fierro - R...|         0|     181173|       0|['Ignacio Corsini']|['5LiOoJbxVSAMkBS...|  1922-03-29|       0.548|0.0391|  6| -23.228|   1|      0.153|       0.996|           0.933|   0.148|  0.612| 75.595|             3|
    |0OYGe21oScKJfanLy...|Capítulo 2.8 - Ba...|         0|      99100|       0|['Fernando Pessoa']|['14jtPCOoNZwquk5...|  1922-06-01|       0.676| 0.235| 11| -22.447|   0|       0.96|       0.794|             0.0|    0.21|  0.724| 96.777|             3|
    |0PE42H6tslQuyMMiG...|Capítulo 2.25 - B...|         0|     132700|       0|['Fernando Pessoa']|['14jtPCOoNZwquk5...|  1922-06-01|        0.75| 0.229|  2| -22.077|   1|      0.955|       0.578|             0.0|   0.314|  0.531|102.629|             3|
    |0PH9AACae1f957JAa...|            Lazy Boi|         0|     157333|       0|            ['Uli']|['45tIt06XoI0Iio4...|  1922-02-22|       0.298|  0.46|  1| -18.645|   1|      0.453|       0.521|           0.856|   0.436|  0.402| 87.921|             4|
    |0QiT0Oo5QdLXdFw6R...|Tu Verras Montmartre|         1|     186800|       0|   ['Lucien Boyer']|['4mSouLpNSEY1d7O...|        1922|       0.703|  0.28|  0|  -15.39|   1|      0.174|       0.995|        6.84e-05|   0.163|  0.897|127.531|             4|
    |0TWsNj5iSvbMTtbED...|Elle Prend L'boul...|         0|     172027|       0|    ['Félix Mayol']|['7DIlOK9L8d0IQ7X...|        1922|       0.709| 0.289|  2| -14.978|   1|       0.18|       0.967|             0.0|   0.119|   0.84|107.515|             4|
    |0cC9CYjLRIzwchQ42...|Capítulo 1.23 - B...|         0|      96600|       0|['Fernando Pessoa']|['14jtPCOoNZwquk5...|  1922-06-01|       0.687| 0.198|  4| -24.264|   0|      0.962|       0.754|             0.0|   0.197|  0.478| 78.453|             1|
    |0eb1PfHxT6HnXvvdU...|Capítulo 1.18 - B...|         0|     103200|       0|['Fernando Pessoa']|['14jtPCOoNZwquk5...|  1922-06-01|         0.8| 0.171|  8| -24.384|   1|      0.953|        0.67|             0.0|   0.123|  0.693| 59.613|             3|
    |0grXU6GKVNCVMJbse...|Capítulo 1.10 - B...|         0|      95800|       0|['Fernando Pessoa']|['14jtPCOoNZwquk5...|  1922-06-01|         0.7| 0.208|  2| -23.874|   1|      0.956|       0.691|             0.0|   0.441|  0.613| 85.739|             1|
    |0kCB1bDVBC8gWCFcn...|  Ca C'est Une Chose|         0|     188000|       0| ['Victor Boucher']|['7vVR02JJYvsEAEP...|        1922|       0.352| 0.334|  5| -13.038|   1|     0.0594|       0.996|         0.00746|    0.36|  0.414| 76.403|             4|
    |0l3BQsVJ7F76wlN5Q...|El Vendaval - Rem...|         0|     153533|       0|['Ignacio Corsini']|['5LiOoJbxVSAMkBS...|  1922-03-21|        0.37| 0.372|  2| -17.138|   1|     0.0865|       0.985|        0.000681|   0.929|  0.753|159.669|             4|
    +--------------------+--------------------+----------+-----------+--------+-------------------+--------------------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-------+--------------+
    only showing top 20 rows
    
    root
     |-- id: string (nullable = true)
     |-- name: string (nullable = true)
     |-- popularity: string (nullable = true)
     |-- duration_ms: string (nullable = true)
     |-- explicit: string (nullable = true)
     |-- artists: string (nullable = true)
     |-- id_artists: string (nullable = true)
     |-- release_date: string (nullable = true)
     |-- danceability: string (nullable = true)
     |-- energy: string (nullable = true)
     |-- key: string (nullable = true)
     |-- loudness: string (nullable = true)
     |-- mode: string (nullable = true)
     |-- speechiness: string (nullable = true)
     |-- acousticness: string (nullable = true)
     |-- instrumentalness: string (nullable = true)
     |-- liveness: string (nullable = true)
     |-- valence: string (nullable = true)
     |-- tempo: string (nullable = true)
     |-- time_signature: string (nullable = true)
    
    




    586672



## Data Cleaning
Both datasets have been cleaned of null values by deleting the records that presented any of them, given their small number of them compared to the total number of records. In second instance, we observed that the tracks dataset was presenting unexpected data for some records in the attribute "Explicit"


```python

tracks_df.select([count(when(isnan(c), c)).alias(c) for c in tracks_df.columns]).show()
artist_df.select([count(when(isnan(c), c)).alias(c) for c in artist_df.columns]).show()
```

    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    | id|name|popularity|duration_ms|explicit|artists|id_artists|release_date|danceability|energy|key|loudness|mode|speechiness|acousticness|instrumentalness|liveness|valence|tempo|time_signature|
    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    |  0|   0|         0|          0|       0|      0|         0|           0|           0|     0|  0|       0|   0|          0|           0|               0|       0|      0|    0|             0|
    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    
    +---+---------+------+----+----------+
    | id|followers|genres|name|popularity|
    +---+---------+------+----+----------+
    |  0|        0|     0|   0|         0|
    +---+---------+------+----+----------+
    
    


```python
#null values
tracks_df.select([count(when(col(c).isNull(), c)).alias(c) for c in tracks_df.columns]).show()
artist_df.select([count(when(col(c).isNull(), c)).alias(c) for c in artist_df.columns]).show()
```

    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    | id|name|popularity|duration_ms|explicit|artists|id_artists|release_date|danceability|energy|key|loudness|mode|speechiness|acousticness|instrumentalness|liveness|valence|tempo|time_signature|
    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    |  0|  71|         0|          0|       0|      0|        12|          12|          12|    12| 12|      12|  12|         12|          12|              12|      12|     12|   12|            12|
    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    
    +---+---------+------+----+----------+
    | id|followers|genres|name|popularity|
    +---+---------+------+----+----------+
    |  0|       13|     0|   0|         0|
    +---+---------+------+----+----------+
    
    

### Casting the Columns Types


```python
artist_df = artist_df.withColumn("followers", artist_df.followers.cast(IntegerType())) \
         .withColumn("popularity", artist_df.popularity.cast(IntegerType()))
```


```python
tracks_df = tracks_df.withColumn("duration_ms", tracks_df.duration_ms.cast(IntegerType())) \
         .withColumn("popularity", tracks_df.popularity.cast(IntegerType())) \
         .withColumn("explicit", tracks_df.explicit.cast(IntegerType())) \
         .withColumn("release_date", tracks_df.release_date.cast(DateType())) \
         .withColumn("danceability", tracks_df.danceability.cast(FloatType())) \
         .withColumn("energy", tracks_df.energy.cast(FloatType())) \
         .withColumn("key", tracks_df.key.cast(IntegerType())) \
         .withColumn("loudness", tracks_df.loudness.cast(FloatType())) \
         .withColumn("mode", tracks_df.mode.cast(IntegerType())) \
         .withColumn("speechiness", tracks_df.speechiness.cast(FloatType())) \
         .withColumn("acousticness", tracks_df.acousticness.cast(FloatType())) \
         .withColumn("instrumentalness", tracks_df.instrumentalness.cast(FloatType())) \
         .withColumn("liveness", tracks_df.liveness.cast(FloatType())) \
         .withColumn("valence", tracks_df.valence.cast(FloatType())) \
         .withColumn("tempo", tracks_df.tempo.cast(FloatType())) \
         .withColumn("time_signature", tracks_df.time_signature.cast(IntegerType()))  
```

After the casting, null values increased, cause some values were not convertible


```python

tracks_df.select([count(when(col(c).isNull(), c)).alias(c) for c in tracks_df.columns]).show()
artist_df.select([count(when(col(c).isNull(), c)).alias(c) for c in artist_df.columns]).show()
```

    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    | id|name|popularity|duration_ms|explicit|artists|id_artists|release_date|danceability|energy|key|loudness|mode|speechiness|acousticness|instrumentalness|liveness|valence|tempo|time_signature|
    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    |  0|  71|      1854|        826|     416|      0|        12|        3309|        2286|   859|430|     268| 160|         91|          64|              45|      34|     27|   23|            55|
    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    
    +---+---------+------+----+----------+
    | id|followers|genres|name|popularity|
    +---+---------+------+----+----------+
    |  0|       13|     0|   0|       531|
    +---+---------+------+----+----------+
    
    


```python
tracks_df = tracks_df.filter(col("release_date").isNotNull())
artist_df= artist_df.withColumn('popularity', coalesce(artist_df['popularity'], lit(0))) \
                         .withColumn('followers', coalesce(artist_df['followers'], lit(0)))

```


```python
#null values
tracks_df.select([count(when(col(c).isNull(), c)).alias(c) for c in tracks_df.columns]).show()
artist_df.select([count(when(col(c).isNull(), c)).alias(c) for c in artist_df.columns]).show()
```

    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    | id|name|popularity|duration_ms|explicit|artists|id_artists|release_date|danceability|energy|key|loudness|mode|speechiness|acousticness|instrumentalness|liveness|valence|tempo|time_signature|
    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    |  0|  71|         0|          0|       0|      0|         0|           0|           0|     0|  0|       0|   0|          0|           0|               0|       0|      0|    0|             0|
    +---+----+----------+-----------+--------+-------+----------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+
    
    +---+---------+------+----+----------+
    | id|followers|genres|name|popularity|
    +---+---------+------+----+----------+
    |  0|        0|     0|   0|         0|
    +---+---------+------+----+----------+
    
    


```python
import datetime

tracks_df= tracks_df.withColumn('age',datediff(current_date(), tracks_df.release_date)/365)
```


```python
tracks_df.select("explicit").distinct().show(500)
```

    +--------+
    |explicit|
    +--------+
    |       1|
    |       0|
    +--------+
    
    

## Data Integration


```python
artist_df = artist_df.withColumn(
    "genres",
    split(regexp_replace(col("genres"), r"(^\[)|(\]$)|(')", ""), ", ")
)
```


```python
tracks_df_wk0= tracks_df.withColumn(
    "id_artists",
    split(regexp_replace(col("id_artists"), r"(^\[)|(\]$)|(')", ""), ", ")
)
tracks_df_wk0
```




    DataFrame[id: string, name: string, popularity: int, duration_ms: int, explicit: int, artists: string, id_artists: array<string>, release_date: date, danceability: float, energy: float, key: int, loudness: float, mode: int, speechiness: float, acousticness: float, instrumentalness: float, liveness: float, valence: float, tempo: float, time_signature: int, age: double]




```python
windowSpec = Window.partitionBy("id_track") 
```


```python
tracks_df_wk1 = tracks_df_wk0.select(col("id").alias("id_track"), "duration_ms", col("popularity").alias("popularity_track"),"explicit", explode(tracks_df_wk0.id_artists).alias("id_artist"),"release_date","danceability","energy","key","loudness","mode", "speechiness","acousticness","instrumentalness","liveness","valence","tempo","time_signature","age")



tracks_df_wk2 = tracks_df_wk1.join(artist_df, tracks_df_wk1.id_artist==artist_df.id,"left") \
           .filter(col("popularity").isNotNull()) \
           .filter(col("followers").isNotNull()) \
           .withColumn("sum_artist_followers",sum(col("followers")).over(windowSpec)) \
           .withColumn("sum_artist_popularity",sum(col("popularity")).over(windowSpec)) \
           .withColumn("avg_artist_followers",F.avg(col("followers")).over(windowSpec)) \
           .withColumn("avg_artist_popularity",F.avg(col("popularity")).over(windowSpec)) \
           .withColumn("collect_list_genres", collect_list("genres").over(windowSpec)) \
           .withColumn("collect_list_genres", flatten(col("collect_list_genres"))) \
           .withColumn("collect_list_genres", array_distinct("collect_list_genres")) \
           .withColumn("genres", array_remove("collect_list_genres", "")) \
           .drop("collect_list_genres") \
           .select("id_track", "popularity_track",  "duration_ms", "genres", "release_date","danceability","energy","key","loudness","mode", "speechiness","acousticness","instrumentalness","liveness","valence","tempo","time_signature", "sum_artist_followers", "sum_artist_popularity","avg_artist_followers","avg_artist_popularity","age").distinct()


```


```python
tracks_df_wk2.select("genres").distinct().show(10, truncate=False)
```

    +----------------------------------------------------------------------------------+
    |genres                                                                            |
    +----------------------------------------------------------------------------------+
    |[mariachi, ranchera]                                                              |
    |[chanson, french jazz, french pop]                                                |
    |[czech folk, czech rock]                                                          |
    |[downtempo, new age]                                                              |
    |[colombian rock, latin, latin alternative, latin pop, latin rock, rock en espanol]|
    |[peruvian rock, pop peruano, pop reggaeton]                                       |
    |[adult standards, vocal jazz]                                                     |
    |[irish country, irish folk]                                                       |
    |[anime]                                                                           |
    |[afrobeat, classic soul, funk, jazz funk, psychedelic soul, soul, southern soul]  |
    +----------------------------------------------------------------------------------+
    only showing top 10 rows
    
    


```python
tracks_df_wk2.printSchema()
```

    root
     |-- id_track: string (nullable = true)
     |-- popularity_track: integer (nullable = true)
     |-- duration_ms: integer (nullable = true)
     |-- genres: array (nullable = false)
     |    |-- element: string (containsNull = true)
     |-- release_date: date (nullable = true)
     |-- danceability: float (nullable = true)
     |-- energy: float (nullable = true)
     |-- key: integer (nullable = true)
     |-- loudness: float (nullable = true)
     |-- mode: integer (nullable = true)
     |-- speechiness: float (nullable = true)
     |-- acousticness: float (nullable = true)
     |-- instrumentalness: float (nullable = true)
     |-- liveness: float (nullable = true)
     |-- valence: float (nullable = true)
     |-- tempo: float (nullable = true)
     |-- time_signature: integer (nullable = true)
     |-- sum_artist_followers: long (nullable = true)
     |-- sum_artist_popularity: long (nullable = true)
     |-- avg_artist_followers: double (nullable = true)
     |-- avg_artist_popularity: double (nullable = true)
     |-- age: double (nullable = true)
    
    


```python
tracks_df_wk2.select([count(when(col(c).isNull(), c)).alias(c) for c in tracks_df_wk2.columns]).show()
```

    +--------+----------------+-----------+------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+--------------------+---------------------+--------------------+---------------------+---+
    |id_track|popularity_track|duration_ms|genres|release_date|danceability|energy|key|loudness|mode|speechiness|acousticness|instrumentalness|liveness|valence|tempo|time_signature|sum_artist_followers|sum_artist_popularity|avg_artist_followers|avg_artist_popularity|age|
    +--------+----------------+-----------+------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+--------------------+---------------------+--------------------+---------------------+---+
    |       0|               0|          0|     0|           0|           0|     0|  0|       0|   0|          0|           0|               0|       0|      0|    0|             0|                   0|                    0|                   0|                    0|  0|
    +--------+----------------+-----------+------+------------+------------+------+---+--------+----+-----------+------------+----------------+--------+-------+-----+--------------+--------------------+---------------------+--------------------+---------------------+---+
    
    


```python
df = tracks_df_wk2
```

# dataset saving


```python
df.write.mode("overwrite").parquet('../data/cleanedDataset_parquet/')
```


```python
df_filtered = df.filter(col('popularity_track')>0)
```


```python
df_filtered.write.mode("overwrite").parquet('../data/cleanedDatasetFiltered_parquet/')
```

# Correlation


```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
```


```python
from pyspark.sql import functions as f
```

### Normalization


```python
columns_to_scale = ["popularity_track",  "duration_ms", "danceability","energy", "loudness", "speechiness","acousticness","instrumentalness","liveness","valence","tempo","time_signature", "avg_artist_followers", "avg_artist_popularity",  "sum_artist_followers", "sum_artist_popularity","age"]
assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in columns_to_scale]
scalers = [MinMaxScaler(inputCol=col + "_vec", outputCol=col + "_scaled") for col in columns_to_scale]
pipeline = Pipeline(stages=assemblers + scalers)
scalerModel = pipeline.fit(df)
enriched_df = scalerModel.transform(df)
```


```python

names = {x + "_scaled": x for x in columns_to_scale}
scaledData = enriched_df.select([f.col(c).alias(names[c]) for c in names.keys()])
```


```python
scaledData.show()
```

    +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+---------------------+--------------------+---------------------+--------------------+
    |    popularity_track|         duration_ms|        danceability|              energy|            loudness|         speechiness|        acousticness|    instrumentalness|            liveness|             valence|               tempo|      time_signature|avg_artist_followers|avg_artist_popularity|sum_artist_followers|sum_artist_popularity|                 age|
    +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+---------------------+--------------------+---------------------+--------------------+
    |               [0.0]|[0.3012114547246877]|[0.6125126088483968]|[0.48399999737739...|[0.7306350931069846]|[0.9876415817464185]|[0.9939759212792594]|[2.70000000455183...|[0.3840000033378601]| [0.453000009059906]|[0.28224174102339...|               [0.8]|[2.48288237015875...|               [0.13]|[4.47946924791007...| [0.01179673321234...|[0.6825733634311512]|
    |              [0.58]|[0.03591518784508...|[0.7991927329728145]|[0.7639999985694885]|[0.8077887910139419]|[0.26158599792998...|[0.10441767368694...|               [0.0]|[0.09560000151395...| [0.531000018119812]|[0.5479156428585806]|               [0.8]|[0.00637347919652...|               [0.63]|[0.00383288450098...| [0.01905626134301...|[0.01106094808126...|
    |              [0.07]|[0.04171738276792...|[0.5227043475483696]|[0.9480000138282776]|[0.8498684522133216]|[0.1266735338868974]|[0.01947791237961...|[1.03999998827930...|[0.5419999957084656]|[0.5410000085830688]|[0.36755674877259...|               [0.8]|[2.79264064033067...|               [0.34]|[1.67943892136209...| [0.01028433151845...|[0.25182844243792...|
    |              [0.21]|[0.04525359593326...|[0.7053481365368595]| [0.746999979019165]|[0.7782672566515104]|[0.04737384072497...|[0.3102409517784656]|[0.00130000000353...|[0.08209999650716...|[0.9049999713897705]|[0.6486538994508209]|               [0.8]|[5.58299991860607...|               [0.45]|[3.35750587664520...| [0.01361161524500...|[0.21252821670428...|
    |              [0.12]|[0.03870788130883...|[0.41977801209306...|[0.7210000157356262]|[0.7336331367223698]|[0.06168898014490...|[0.6104417524076896]|[0.00559999980032...|[0.5960000157356262]|[0.8579999804496765]|[0.5561062314108192]|               [0.8]|[2.51241333454093...|               [0.21]|[1.51091575465920...| [0.00635208711433...| [0.357607223476298]|
    |               [0.1]|[0.06785467242590...|[0.3400605585598796]|[0.7419999837875366]|[0.8217541622178045]|[0.02996910441475...|[0.5622490029537214]| [0.781000018119812]|[0.15000000596046...|[0.25600001215934...|[0.4588056785157454]|               [0.8]|[1.94498789445922...|               [0.28]|[1.16967730267871...| [0.00846944948578...|[0.3486681715575621]|
    |              [0.34]|[0.04405296380801...|[0.5761856527710055]|[0.5720000267028809]|[0.8138307633182191]|[0.03171987573014...|[0.4638554379095239]|               [0.0]|[0.25099998712539...| [0.460999995470047]|[0.6093814108965695]|               [0.8]|[1.58047693496067...|               [0.27]|[9.50467611390821...| [0.00816696914700...|[0.1590293453724605]|
    |               [0.0]|[0.02455786655236...|[0.7013118014009317]|[0.11900000274181...|[0.6201205398182331]|[0.23069001368083...|[0.9939759212792594]|[1.90999999176710...|[0.10400000214576...|[0.6930000185966492]|[0.5279506032919785]|               [0.8]|[4.47146962834102...|               [0.23]|[2.68905351482503...| [0.00695704779189...|[0.7488036117381489]|
    |[0.47000000000000...|[0.06235864314507...|[0.6064581362174845]|[0.7889999747276306]|[0.7842327538332763]|[0.0629248173663992]|[5.98393577200014...|[0.10400000214576...|[0.24799999594688...|[0.9229999780654907]|[0.6350286821093312]|               [0.8]|[4.30670459101553...|               [0.37]|[2.58996707578669...| [0.01119177253478...|[0.3321670428893905]|
    |              [0.37]|[0.02724927614966...|[0.34409686362282...|[0.9639999866485596]|[0.8362854857970745]|[0.11019567357452...|[1.26506029600617...|[1.02999998489394...|[0.1509999930858612]|[0.45899999141693...|[0.6177099832319977]|               [0.8]|[0.15852036636545...|               [0.84]|[0.09533101726658...| [0.02540834845735...|[0.24367945823927...|
    |               [0.3]|[0.08530629202434...|[0.9192734777193206]|[0.5370000004768372]|[0.8223201218869766]|[0.08032955751317...|[0.01174698772874...|[0.00255999993532...|[0.04340000078082...|[0.0934000015258789]| [0.507311839943766]|               [0.8]|[0.00833901202371...|              [0.625]|[0.01002983424081...| [0.03781004234724...|[0.09970654627539...|
    |              [0.38]|[0.05633732618424...|[0.21897073002939...|[0.13899999856948...| [0.705549444069622]|[0.03903192605193...|[0.9588353324128924]|[0.3059999942779541]|[0.08420000225305...|[0.11900000274181...|[0.47711067036398...|               [0.8]|[0.00116458463228...|               [0.65]|[7.00358195190513...| [0.01966122202056...|[0.1590293453724605]|
    |              [0.13]|[0.02120428475255...|[0.4217961796610306]|[0.9580000042915344]| [0.827964394379136]|[0.0514933058030676]|[0.00108433731621...|               [0.0]|[0.23899999260902...|[0.7310000061988831]|[0.5059075390982438]|               [0.8]|[0.00140851293292...|               [0.55]|[8.47051857169886...| [0.01663641863278...| [0.208510158013544]|
    |              [0.05]|[0.02436562300970...|[0.5176589436649495]|[0.3970000147819519]|[0.7620686514074935]|[0.07188465065018...|[0.9718875528445151]|[3.23999993270263...|[0.1860000044107437]|[0.6899999976158142]|[0.29415417582094...|[0.6000000000000001]|[0.00139877912149...|               [0.45]|[8.41198135232852...| [0.01361161524500...|[0.5520767494356658]|
    |              [0.59]|[0.07461292296694...|               [0.0]|[0.01630000025033...| [0.544251704200878]|               [0.0]|[0.7309236858866315]|[0.32600000500679...|[0.11100000143051...|               [0.0]|               [0.0]|               [0.0]|[1.53979264497491...|               [0.62]|[9.26000882982124...| [0.01875378100423...|[0.09641083521444...|
    |              [0.18]|[0.0398946291782265]|[0.5701311801400932]|[0.5429999828338623]|[0.7122950385573563]|[0.02543769359615...|[0.3955823263130512]|[0.00781999994069...|[0.20600000023841...|[0.7210000157356262]|[0.40494193130847...|               [0.8]|[1.87325173205443...|               [0.24]|[1.12653659152817...| [0.00725952813067...|[0.31568848758465...|
    |              [0.49]|[0.02812807834422...|[0.4419777951947108]|[0.8080000281333923]|[0.8287291966606023]|[0.34912459439224...|[0.00159638555501...|               [0.0]|[0.14499999582767...| [0.746999979019165]|[0.7302876571709919]|               [0.8]|[0.0053990207430817]| [0.6900000000000001]|[0.00324686443440...| [0.02087114337568...|[0.04092550790067...|
    |               [0.5]|[0.05444105723980...|[0.7295660872064674]| [0.703000009059906]|[0.7725312067185726]|[0.03274974391794...|[0.02730923753211...|[0.31299999356269...|[0.09790000319480...|[0.7670000195503235]|[0.5265219632961898]|               [0.8]|[0.00371706121936...|               [0.55]|[0.00223536719860...| [0.01663641863278...|[0.14923250564334...|
    |              [0.31]|[0.03264597960011...|[0.6710393781004114]|[0.7910000085830688]|[0.8340828418059247]|[0.03120494355452...|[0.03142570391679...|[1.82999996468424...|[0.3449999988079071]| [0.703000009059906]|[0.4383414386419621]|               [0.8]|[1.43599067145985...|              [0.395]|[1.72715285277594...| [0.02389594676346...|[0.02467268623024...|
    |              [0.14]|[0.08304013226355...|[0.2734611491089888]|[0.8479999899864197]|[0.7588258728125425]|[0.07981462341927...|[0.00442771093192...|[0.6299999952316284]|[0.11299999803304...|[0.38499999046325...|[0.6436332220111385]|               [0.8]|[2.99111913913968...|                [0.3]|[1.79879997023443...| [0.00907441016333...| [0.381647855530474]|
    +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+---------------------+--------------------+---------------------+--------------------+
    only showing top 20 rows
    
    


```python
scaledData.printSchema()
```

    root
     |-- popularity_track: vector (nullable = true)
     |-- duration_ms: vector (nullable = true)
     |-- danceability: vector (nullable = true)
     |-- energy: vector (nullable = true)
     |-- loudness: vector (nullable = true)
     |-- speechiness: vector (nullable = true)
     |-- acousticness: vector (nullable = true)
     |-- instrumentalness: vector (nullable = true)
     |-- liveness: vector (nullable = true)
     |-- valence: vector (nullable = true)
     |-- tempo: vector (nullable = true)
     |-- time_signature: vector (nullable = true)
     |-- avg_artist_followers: vector (nullable = true)
     |-- avg_artist_popularity: vector (nullable = true)
     |-- sum_artist_followers: vector (nullable = true)
     |-- sum_artist_popularity: vector (nullable = true)
     |-- age: vector (nullable = true)
    
    

CORRELATION


```python
from pyspark.ml.stat import Correlation
```


```python
import pandas as pd
```


```python
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=scaledData.columns, outputCol=vector_col)
df_vector = assembler.transform(scaledData).select(vector_col)


matrix = Correlation.corr(df_vector, vector_col)
corrmatrix = matrix.collect()[0]["pearson({})".format(vector_col)].values
```


```python
pd.DataFrame(corrmatrix.reshape(-1, len(scaledData.columns)), columns=scaledData.columns, index=scaledData.columns)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity_track</th>
      <th>duration_ms</th>
      <th>danceability</th>
      <th>energy</th>
      <th>loudness</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>avg_artist_followers</th>
      <th>avg_artist_popularity</th>
      <th>sum_artist_followers</th>
      <th>sum_artist_popularity</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>popularity_track</th>
      <td>1.000000</td>
      <td>0.036382</td>
      <td>0.187216</td>
      <td>0.308021</td>
      <td>0.332595</td>
      <td>-0.050129</td>
      <td>-0.379428</td>
      <td>-0.236680</td>
      <td>-0.049918</td>
      <td>-0.003150</td>
      <td>0.071875</td>
      <td>0.089161</td>
      <td>0.239132</td>
      <td>0.560317</td>
      <td>0.238146</td>
      <td>0.289265</td>
      <td>-0.609580</td>
    </tr>
    <tr>
      <th>duration_ms</th>
      <td>0.036382</td>
      <td>1.000000</td>
      <td>-0.125574</td>
      <td>0.023709</td>
      <td>-0.000336</td>
      <td>-0.135599</td>
      <td>-0.065325</td>
      <td>0.070266</td>
      <td>0.001978</td>
      <td>-0.167480</td>
      <td>-0.000753</td>
      <td>0.040369</td>
      <td>0.019331</td>
      <td>0.006293</td>
      <td>0.028669</td>
      <td>0.074758</td>
      <td>-0.056375</td>
    </tr>
    <tr>
      <th>danceability</th>
      <td>0.187216</td>
      <td>-0.125574</td>
      <td>1.000000</td>
      <td>0.235468</td>
      <td>0.244430</td>
      <td>0.199716</td>
      <td>-0.235509</td>
      <td>-0.231624</td>
      <td>-0.105230</td>
      <td>0.526467</td>
      <td>-0.048310</td>
      <td>0.141694</td>
      <td>0.021801</td>
      <td>0.039417</td>
      <td>0.036300</td>
      <td>-0.017918</td>
      <td>-0.224180</td>
    </tr>
    <tr>
      <th>energy</th>
      <td>0.308021</td>
      <td>0.023709</td>
      <td>0.235468</td>
      <td>1.000000</td>
      <td>0.764760</td>
      <td>-0.054952</td>
      <td>-0.714489</td>
      <td>-0.201543</td>
      <td>0.125767</td>
      <td>0.369151</td>
      <td>0.227515</td>
      <td>0.187793</td>
      <td>0.094059</td>
      <td>0.168460</td>
      <td>0.089817</td>
      <td>0.007919</td>
      <td>-0.462166</td>
    </tr>
    <tr>
      <th>loudness</th>
      <td>0.332595</td>
      <td>-0.000336</td>
      <td>0.244430</td>
      <td>0.764760</td>
      <td>1.000000</td>
      <td>-0.170993</td>
      <td>-0.518274</td>
      <td>-0.331347</td>
      <td>0.029592</td>
      <td>0.268675</td>
      <td>0.186477</td>
      <td>0.162221</td>
      <td>0.120709</td>
      <td>0.144123</td>
      <td>0.117405</td>
      <td>-0.024699</td>
      <td>-0.454457</td>
    </tr>
    <tr>
      <th>speechiness</th>
      <td>-0.050129</td>
      <td>-0.135599</td>
      <td>0.199716</td>
      <td>-0.054952</td>
      <td>-0.170993</td>
      <td>1.000000</td>
      <td>0.070266</td>
      <td>-0.101961</td>
      <td>0.208763</td>
      <td>0.045535</td>
      <td>-0.089222</td>
      <td>-0.116968</td>
      <td>-0.029310</td>
      <td>0.052484</td>
      <td>-0.020968</td>
      <td>0.052302</td>
      <td>0.082275</td>
    </tr>
    <tr>
      <th>acousticness</th>
      <td>-0.379428</td>
      <td>-0.065325</td>
      <td>-0.235509</td>
      <td>-0.714489</td>
      <td>-0.518274</td>
      <td>0.070266</td>
      <td>1.000000</td>
      <td>0.215057</td>
      <td>-0.006041</td>
      <td>-0.176285</td>
      <td>-0.192577</td>
      <td>-0.173623</td>
      <td>-0.119497</td>
      <td>-0.219779</td>
      <td>-0.111970</td>
      <td>-0.030148</td>
      <td>0.526501</td>
    </tr>
    <tr>
      <th>instrumentalness</th>
      <td>-0.236680</td>
      <td>0.070266</td>
      <td>-0.231624</td>
      <td>-0.201543</td>
      <td>-0.331347</td>
      <td>-0.101961</td>
      <td>0.215057</td>
      <td>1.000000</td>
      <td>-0.037321</td>
      <td>-0.170356</td>
      <td>-0.056607</td>
      <td>-0.042552</td>
      <td>-0.055176</td>
      <td>-0.124316</td>
      <td>-0.048134</td>
      <td>0.023272</td>
      <td>0.244716</td>
    </tr>
    <tr>
      <th>liveness</th>
      <td>-0.049918</td>
      <td>0.001978</td>
      <td>-0.105230</td>
      <td>0.125767</td>
      <td>0.029592</td>
      <td>0.208763</td>
      <td>-0.006041</td>
      <td>-0.037321</td>
      <td>1.000000</td>
      <td>0.000399</td>
      <td>-0.014151</td>
      <td>-0.023823</td>
      <td>0.006899</td>
      <td>0.050420</td>
      <td>0.001481</td>
      <td>0.026381</td>
      <td>0.018199</td>
    </tr>
    <tr>
      <th>valence</th>
      <td>-0.003150</td>
      <td>-0.167480</td>
      <td>0.526467</td>
      <td>0.369151</td>
      <td>0.268675</td>
      <td>0.045535</td>
      <td>-0.176285</td>
      <td>-0.170356</td>
      <td>0.000399</td>
      <td>1.000000</td>
      <td>0.131939</td>
      <td>0.102432</td>
      <td>-0.035995</td>
      <td>-0.056572</td>
      <td>-0.034816</td>
      <td>-0.108460</td>
      <td>0.026984</td>
    </tr>
    <tr>
      <th>tempo</th>
      <td>0.071875</td>
      <td>-0.000753</td>
      <td>-0.048310</td>
      <td>0.227515</td>
      <td>0.186477</td>
      <td>-0.089222</td>
      <td>-0.192577</td>
      <td>-0.056607</td>
      <td>-0.014151</td>
      <td>0.131939</td>
      <td>1.000000</td>
      <td>0.027672</td>
      <td>0.015348</td>
      <td>0.013545</td>
      <td>0.011706</td>
      <td>-0.034763</td>
      <td>-0.118860</td>
    </tr>
    <tr>
      <th>time_signature</th>
      <td>0.089161</td>
      <td>0.040369</td>
      <td>0.141694</td>
      <td>0.187793</td>
      <td>0.162221</td>
      <td>-0.116968</td>
      <td>-0.173623</td>
      <td>-0.042552</td>
      <td>-0.023823</td>
      <td>0.102432</td>
      <td>0.027672</td>
      <td>1.000000</td>
      <td>0.028661</td>
      <td>0.027769</td>
      <td>0.028969</td>
      <td>-0.004478</td>
      <td>-0.117001</td>
    </tr>
    <tr>
      <th>avg_artist_followers</th>
      <td>0.239132</td>
      <td>0.019331</td>
      <td>0.021801</td>
      <td>0.094059</td>
      <td>0.120709</td>
      <td>-0.029310</td>
      <td>-0.119497</td>
      <td>-0.055176</td>
      <td>0.006899</td>
      <td>-0.035995</td>
      <td>0.015348</td>
      <td>0.028661</td>
      <td>1.000000</td>
      <td>0.424686</td>
      <td>0.919003</td>
      <td>0.229386</td>
      <td>-0.127868</td>
    </tr>
    <tr>
      <th>avg_artist_popularity</th>
      <td>0.560317</td>
      <td>0.006293</td>
      <td>0.039417</td>
      <td>0.168460</td>
      <td>0.144123</td>
      <td>0.052484</td>
      <td>-0.219779</td>
      <td>-0.124316</td>
      <td>0.050420</td>
      <td>-0.056572</td>
      <td>0.013545</td>
      <td>0.027769</td>
      <td>0.424686</td>
      <td>1.000000</td>
      <td>0.402551</td>
      <td>0.536890</td>
      <td>-0.305300</td>
    </tr>
    <tr>
      <th>sum_artist_followers</th>
      <td>0.238146</td>
      <td>0.028669</td>
      <td>0.036300</td>
      <td>0.089817</td>
      <td>0.117405</td>
      <td>-0.020968</td>
      <td>-0.111970</td>
      <td>-0.048134</td>
      <td>0.001481</td>
      <td>-0.034816</td>
      <td>0.011706</td>
      <td>0.028969</td>
      <td>0.919003</td>
      <td>0.402551</td>
      <td>1.000000</td>
      <td>0.370236</td>
      <td>-0.137776</td>
    </tr>
    <tr>
      <th>sum_artist_popularity</th>
      <td>0.289265</td>
      <td>0.074758</td>
      <td>-0.017918</td>
      <td>0.007919</td>
      <td>-0.024699</td>
      <td>0.052302</td>
      <td>-0.030148</td>
      <td>0.023272</td>
      <td>0.026381</td>
      <td>-0.108460</td>
      <td>-0.034763</td>
      <td>-0.004478</td>
      <td>0.229386</td>
      <td>0.536890</td>
      <td>0.370236</td>
      <td>1.000000</td>
      <td>-0.149582</td>
    </tr>
    <tr>
      <th>age</th>
      <td>-0.609580</td>
      <td>-0.056375</td>
      <td>-0.224180</td>
      <td>-0.462166</td>
      <td>-0.454457</td>
      <td>0.082275</td>
      <td>0.526501</td>
      <td>0.244716</td>
      <td>0.018199</td>
      <td>0.026984</td>
      <td>-0.118860</td>
      <td>-0.117001</td>
      <td>-0.127868</td>
      <td>-0.305300</td>
      <td>-0.137776</td>
      <td>-0.149582</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Correlation and Normalization Filtered dataset


```python
columns_to_scale = ["popularity_track",  "duration_ms", "danceability","energy", "loudness", "speechiness","acousticness","instrumentalness","liveness","valence","tempo","time_signature", "avg_artist_followers", "avg_artist_popularity",  "sum_artist_followers", "sum_artist_popularity","age"]
assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in columns_to_scale]
scalers = [MinMaxScaler(inputCol=col + "_vec", outputCol=col + "_scaled") for col in columns_to_scale]
pipeline = Pipeline(stages=assemblers + scalers)
scalerModel = pipeline.fit(df_filtered)
enriched_df = scalerModel.transform(df_filtered)


names = {x + "_scaled": x for x in columns_to_scale}
scaledData = enriched_df.select([f.col(c).alias(names[c]) for c in names.keys()])


```


```python
scaledData.show()
```

    +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+---------------------+--------------------+---------------------+--------------------+
    |    popularity_track|         duration_ms|        danceability|              energy|            loudness|         speechiness|        acousticness|    instrumentalness|            liveness|             valence|               tempo|      time_signature|avg_artist_followers|avg_artist_popularity|sum_artist_followers|sum_artist_popularity|                 age|
    +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+---------------------+--------------------+---------------------+--------------------+
    |[0.5757575757575758]|[0.03396105598670...|[0.7991927329728145]|[0.7639999985694885]|[0.8077887910139419]|[0.26158599792998...|[0.10441767368694...|               [0.0]|[0.09560000151395...| [0.531000018119812]|[0.5479156428585806]|               [0.8]|[0.00637347919652...|               [0.63]|[0.00383288450098...| [0.01905626134301...|[0.01099397250378...|
    |[0.06060606060606...|[0.0397750115490742]|[0.5227043475483696]|[0.9480000138282776]|[0.8498684522133216]|[0.1266735338868974]|[0.01947791237961...|[1.03999998827930...|[0.5419999957084656]|[0.5410000085830688]|[0.36755674877259...|               [0.8]|[2.79264064033067...|               [0.34]|[1.67943892136209...| [0.01028433151845...|[0.25177777276113...|
    |[0.20202020202020...|[0.04331839236887...|[0.7053481365368595]| [0.746999979019165]|[0.7782672566515104]|[0.04737384072497...|[0.3102409517784656]|[0.00130000000353...|[0.08209999650716...|[0.9049999713897705]|[0.6486538994508209]|               [0.8]|[5.58299991860607...|               [0.45]|[3.35750587664520...| [0.01361161524500...|[0.21247488543242...|
    |[0.11111111111111...|[0.03675941004296...|[0.41977801209306...|[0.7210000157356262]|[0.7336331367223698]|[0.06168898014490...|[0.6104417524076896]|[0.00559999980032...|[0.5960000157356262]|[0.8579999804496765]|[0.5561062314108192]|               [0.8]|[2.51241333454093...|               [0.21]|[1.51091575465920...| [0.00635208711433...|[0.3575637176332483]|
    |[0.09090909090909...|[0.06596527964812...|[0.3400605585598796]|[0.7419999837875366]|[0.8217541622178045]|[0.02996910441475...|[0.5622490029537214]| [0.781000018119812]|[0.15000000596046...|[0.25600001215934...|[0.4588056785157454]|               [0.8]|[1.94498789445922...|               [0.28]|[1.16967730267871...| [0.00846944948578...|[0.34862406032011...|
    |[0.33333333333333...|[0.04211532664705...|[0.5761856527710055]|[0.5720000267028809]|[0.8138307633182191]|[0.03171987573014...|[0.4638554379095239]|               [0.0]|[0.25099998712539...| [0.460999995470047]|[0.6093814108965695]|               [0.8]|[1.58047693496067...|               [0.27]|[9.50467611390821...| [0.00816696914700...|[0.15897239090683...|
    |[0.4646464646464647]|[0.06045811030391...|[0.6064581362174845]|[0.7889999747276306]|[0.7842327538332763]|[0.0629248173663992]|[5.98393577200014...|[0.10400000214576...|[0.24799999594688...|[0.9229999780654907]|[0.6350286821093312]|               [0.8]|[4.30670459101553...|               [0.37]|[2.58996707578669...| [0.01119177253478...|[0.3321218141183376]|
    |[0.36363636363636...|[0.02527757910001...|[0.34409686362282...|[0.9639999866485596]|[0.8362854857970745]|[0.11019567357452...|[1.26506029600617...|[1.02999998489394...|[0.1509999930858612]|[0.45899999141693...|[0.6177099832319977]|               [0.8]|[0.15852036636545...|               [0.84]|[0.09533101726658...| [0.02540834845735...|[0.24362823667516...|
    |[0.29292929292929...|[0.08345227244756...|[0.9192734777193206]|[0.5370000004768372]|[0.8223201218869766]|[0.08032955751317...|[0.01174698772874...|[0.00255999993532...|[0.04340000078082...|[0.0934000015258789]| [0.507311839943766]|               [0.8]|[0.00833901202371...|              [0.625]|[0.01002983424081...| [0.03781004234724...|[0.09964557419238...|
    |[0.37373737373737...|[0.05442458855865...|[0.21897073002939...|[0.13899999856948...| [0.705549444069622]|[0.03903192605193...|[0.9588353324128924]|[0.3059999942779541]|[0.08420000225305...|[0.11900000274181...|[0.47711067036398...|               [0.8]|[0.00116458463228...|               [0.65]|[7.00358195190513...| [0.01966122202056...|[0.15897239090683...|
    |[0.12121212121212...|[0.01922033493207...|[0.4217961796610306]|[0.9580000042915344]| [0.827964394379136]|[0.0514933058030676]|[0.00108433731621...|               [0.0]|[0.23899999260902...|[0.7310000061988831]|[0.5059075390982438]|               [0.8]|[0.00140851293292...|               [0.55]|[8.47051857169886...| [0.01663641863278...|[0.2084565546199517]|
    |[0.04040404040404...|[0.02238808099869...|[0.5176589436649495]|[0.3970000147819519]|[0.7620686514074935]|[0.07188465065018...|[0.9718875528445151]|[3.23999993270263...|[0.1860000044107437]|[0.6899999976158142]|[0.29415417582094...|[0.6000000000000001]|[0.00139877912149...|               [0.45]|[8.41198135232852...| [0.01361161524500...|[0.5520464139783732]|
    |[0.5858585858585859]|[0.07273722868593...|               [0.0]|[0.01630000025033...| [0.544251704200878]|               [0.0]|[0.7309236858866315]|[0.32600000500679...|[0.11100000143051...|               [0.0]|               [0.0]|               [0.0]|[1.53979264497491...|               [0.62]|[9.26000882982124...| [0.01875378100423...|[0.09634963993046...|
    |[0.17171717171717...| [0.037948563366515]|[0.5701311801400932]|[0.5429999828338623]|[0.7122950385573563]|[0.02543769359615...|[0.3955823263130512]|[0.00781999994069...|[0.20600000023841...|[0.7210000157356262]|[0.40494193130847...|               [0.8]|[1.87325173205443...|               [0.24]|[1.12653659152817...| [0.00725952813067...|[0.31564214280876...|
    |[0.48484848484848...|[0.02615816256459...|[0.4419777951947108]|[0.8080000281333923]|[0.8287291966606023]|[0.34912459439224...|[0.00159638555501...|               [0.0]|[0.14499999582767...| [0.746999979019165]|[0.7302876571709919]|               [0.8]|[0.0053990207430817]| [0.6900000000000001]|[0.00324686443440...| [0.02087114337568...|[0.0408605548908504]|
    | [0.494949494949495]|[0.05252447601092...|[0.7295660872064674]| [0.703000009059906]|[0.7725312067185726]|[0.03274974391794...|[0.02730923753211...|[0.31299999356269...|[0.09790000319480...|[0.7670000195503235]|[0.5265219632961898]|               [0.8]|[0.00371706121936...|               [0.55]|[0.00223536719860...| [0.01663641863278...|[0.14917488768991...|
    |[0.30303030303030...|[0.03068522128739...|[0.6710393781004114]|[0.7910000085830688]|[0.8340828418059247]|[0.03120494355452...|[0.03142570391679...|[1.82999996468424...|[0.3449999988079071]| [0.703000009059906]|[0.4383414386419621]|               [0.8]|[1.43599067145985...|              [0.395]|[1.72715285277594...| [0.02389594676346...|[0.02460663250332...|
    |[0.13131313131313...|[0.08118151934090...|[0.2734611491089888]|[0.8479999899864197]|[0.7588258728125425]|[0.07981462341927...|[0.00442771093192...|[0.6299999952316284]|[0.11299999803304...|[0.38499999046325...|[0.6436332220111385]|               [0.8]|[2.99111913913968...|                [0.3]|[1.79879997023443...| [0.00907441016333...|[0.3816059778314558]|
    |[0.29292929292929...|[0.01722979179560...|[0.45105953421405...|[0.9990000128746033]|[0.7767376375010485]|[0.06941297620718...|[0.00158634543471...|[0.01040000002831...|[0.5299999713897705]|[0.45899999141693...|[0.5599051825978173]|               [0.8]|[1.65018521998299...|               [0.48]|[9.92388797137810...| [0.01451905626134...|[0.00277671174120...|
    |[0.24242424242424...|[0.02872821059803...|[0.12815338494538...|[0.36899998784065...|[0.7058859491962012]|[0.04037075162683...|[0.5261044109412346]|[0.9049999713897705]|[0.16500000655651...|[0.19200000166893...|[0.3228373999222758]|               [0.8]|[0.00766708119015...|              [0.765]|[0.00922166237791...| [0.04627949183303...|[0.1809377610221911]|
    +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+---------------------+--------------------+---------------------+--------------------+
    only showing top 20 rows
    
    


```python
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=scaledData.columns, outputCol=vector_col)
df_vector = assembler.transform(scaledData).select(vector_col)


matrix = Correlation.corr(df_vector, vector_col)
corrmatrix = matrix.collect()[0]["pearson({})".format(vector_col)].values
```


```python
pd.DataFrame(corrmatrix.reshape(-1, len(scaledData.columns)), columns=scaledData.columns, index=scaledData.columns)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity_track</th>
      <th>duration_ms</th>
      <th>danceability</th>
      <th>energy</th>
      <th>loudness</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>avg_artist_followers</th>
      <th>avg_artist_popularity</th>
      <th>sum_artist_followers</th>
      <th>sum_artist_popularity</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>popularity_track</th>
      <td>1.000000</td>
      <td>0.033927</td>
      <td>0.183881</td>
      <td>0.250134</td>
      <td>0.305949</td>
      <td>-0.022462</td>
      <td>-0.307831</td>
      <td>-0.165830</td>
      <td>-0.055666</td>
      <td>-0.016078</td>
      <td>0.051712</td>
      <td>0.076138</td>
      <td>0.250309</td>
      <td>0.513139</td>
      <td>0.252914</td>
      <td>0.275534</td>
      <td>-0.557065</td>
    </tr>
    <tr>
      <th>duration_ms</th>
      <td>0.033927</td>
      <td>1.000000</td>
      <td>-0.123365</td>
      <td>0.017221</td>
      <td>-0.002648</td>
      <td>-0.149233</td>
      <td>-0.058292</td>
      <td>0.071600</td>
      <td>0.000515</td>
      <td>-0.157342</td>
      <td>0.001088</td>
      <td>0.040379</td>
      <td>0.020172</td>
      <td>-0.012053</td>
      <td>0.028716</td>
      <td>0.057902</td>
      <td>-0.039230</td>
    </tr>
    <tr>
      <th>danceability</th>
      <td>0.183881</td>
      <td>-0.123365</td>
      <td>1.000000</td>
      <td>0.225898</td>
      <td>0.235053</td>
      <td>0.190714</td>
      <td>-0.227859</td>
      <td>-0.230043</td>
      <td>-0.106630</td>
      <td>0.519523</td>
      <td>-0.068337</td>
      <td>0.140885</td>
      <td>0.014272</td>
      <td>0.032054</td>
      <td>0.031979</td>
      <td>0.000024</td>
      <td>-0.237185</td>
    </tr>
    <tr>
      <th>energy</th>
      <td>0.250134</td>
      <td>0.017221</td>
      <td>0.225898</td>
      <td>1.000000</td>
      <td>0.763668</td>
      <td>-0.033219</td>
      <td>-0.700022</td>
      <td>-0.169866</td>
      <td>0.129641</td>
      <td>0.381403</td>
      <td>0.215175</td>
      <td>0.185575</td>
      <td>0.082891</td>
      <td>0.110304</td>
      <td>0.081107</td>
      <td>-0.010588</td>
      <td>-0.409324</td>
    </tr>
    <tr>
      <th>loudness</th>
      <td>0.305949</td>
      <td>-0.002648</td>
      <td>0.235053</td>
      <td>0.763668</td>
      <td>1.000000</td>
      <td>-0.143297</td>
      <td>-0.518267</td>
      <td>-0.325237</td>
      <td>0.028035</td>
      <td>0.263791</td>
      <td>0.173630</td>
      <td>0.155545</td>
      <td>0.115361</td>
      <td>0.111662</td>
      <td>0.114957</td>
      <td>-0.028688</td>
      <td>-0.449045</td>
    </tr>
    <tr>
      <th>speechiness</th>
      <td>-0.022462</td>
      <td>-0.149233</td>
      <td>0.190714</td>
      <td>-0.033219</td>
      <td>-0.143297</td>
      <td>1.000000</td>
      <td>0.067606</td>
      <td>-0.098750</td>
      <td>0.224581</td>
      <td>0.038608</td>
      <td>-0.089718</td>
      <td>-0.113521</td>
      <td>-0.025477</td>
      <td>0.109719</td>
      <td>-0.016189</td>
      <td>0.090585</td>
      <td>0.036217</td>
    </tr>
    <tr>
      <th>acousticness</th>
      <td>-0.307831</td>
      <td>-0.058292</td>
      <td>-0.227859</td>
      <td>-0.700022</td>
      <td>-0.518267</td>
      <td>0.067606</td>
      <td>1.000000</td>
      <td>0.164367</td>
      <td>-0.006411</td>
      <td>-0.192153</td>
      <td>-0.182970</td>
      <td>-0.174568</td>
      <td>-0.106299</td>
      <td>-0.142260</td>
      <td>-0.101054</td>
      <td>0.001825</td>
      <td>0.447438</td>
    </tr>
    <tr>
      <th>instrumentalness</th>
      <td>-0.165830</td>
      <td>0.071600</td>
      <td>-0.230043</td>
      <td>-0.169866</td>
      <td>-0.325237</td>
      <td>-0.098750</td>
      <td>0.164367</td>
      <td>1.000000</td>
      <td>-0.033033</td>
      <td>-0.177621</td>
      <td>-0.043464</td>
      <td>-0.038565</td>
      <td>-0.044391</td>
      <td>-0.059807</td>
      <td>-0.039885</td>
      <td>0.050047</td>
      <td>0.178743</td>
    </tr>
    <tr>
      <th>liveness</th>
      <td>-0.055666</td>
      <td>0.000515</td>
      <td>-0.106630</td>
      <td>0.129641</td>
      <td>0.028035</td>
      <td>0.224581</td>
      <td>-0.006411</td>
      <td>-0.033033</td>
      <td>1.000000</td>
      <td>-0.000846</td>
      <td>-0.014522</td>
      <td>-0.024734</td>
      <td>0.009899</td>
      <td>0.061700</td>
      <td>0.004407</td>
      <td>0.030925</td>
      <td>0.023298</td>
    </tr>
    <tr>
      <th>valence</th>
      <td>-0.016078</td>
      <td>-0.157342</td>
      <td>0.519523</td>
      <td>0.381403</td>
      <td>0.263791</td>
      <td>0.038608</td>
      <td>-0.192153</td>
      <td>-0.177621</td>
      <td>-0.000846</td>
      <td>1.000000</td>
      <td>0.124335</td>
      <td>0.103256</td>
      <td>-0.040722</td>
      <td>-0.060194</td>
      <td>-0.037706</td>
      <td>-0.093217</td>
      <td>0.032377</td>
    </tr>
    <tr>
      <th>tempo</th>
      <td>0.051712</td>
      <td>0.001088</td>
      <td>-0.068337</td>
      <td>0.215175</td>
      <td>0.173630</td>
      <td>-0.089718</td>
      <td>-0.182970</td>
      <td>-0.043464</td>
      <td>-0.014522</td>
      <td>0.124335</td>
      <td>1.000000</td>
      <td>0.018928</td>
      <td>0.012308</td>
      <td>-0.008809</td>
      <td>0.009221</td>
      <td>-0.041702</td>
      <td>-0.098451</td>
    </tr>
    <tr>
      <th>time_signature</th>
      <td>0.076138</td>
      <td>0.040379</td>
      <td>0.140885</td>
      <td>0.185575</td>
      <td>0.155545</td>
      <td>-0.113521</td>
      <td>-0.174568</td>
      <td>-0.038565</td>
      <td>-0.024734</td>
      <td>0.103256</td>
      <td>0.018928</td>
      <td>1.000000</td>
      <td>0.026134</td>
      <td>0.007469</td>
      <td>0.027032</td>
      <td>-0.011954</td>
      <td>-0.104819</td>
    </tr>
    <tr>
      <th>avg_artist_followers</th>
      <td>0.250309</td>
      <td>0.020172</td>
      <td>0.014272</td>
      <td>0.082891</td>
      <td>0.115361</td>
      <td>-0.025477</td>
      <td>-0.106299</td>
      <td>-0.044391</td>
      <td>0.009899</td>
      <td>-0.040722</td>
      <td>0.012308</td>
      <td>0.026134</td>
      <td>1.000000</td>
      <td>0.440156</td>
      <td>0.918780</td>
      <td>0.227238</td>
      <td>-0.110662</td>
    </tr>
    <tr>
      <th>avg_artist_popularity</th>
      <td>0.513139</td>
      <td>-0.012053</td>
      <td>0.032054</td>
      <td>0.110304</td>
      <td>0.111662</td>
      <td>0.109719</td>
      <td>-0.142260</td>
      <td>-0.059807</td>
      <td>0.061700</td>
      <td>-0.060194</td>
      <td>-0.008809</td>
      <td>0.007469</td>
      <td>0.440156</td>
      <td>1.000000</td>
      <td>0.416371</td>
      <td>0.508247</td>
      <td>-0.181174</td>
    </tr>
    <tr>
      <th>sum_artist_followers</th>
      <td>0.252914</td>
      <td>0.028716</td>
      <td>0.031979</td>
      <td>0.081107</td>
      <td>0.114957</td>
      <td>-0.016189</td>
      <td>-0.101054</td>
      <td>-0.039885</td>
      <td>0.004407</td>
      <td>-0.037706</td>
      <td>0.009221</td>
      <td>0.027032</td>
      <td>0.918780</td>
      <td>0.416371</td>
      <td>1.000000</td>
      <td>0.373577</td>
      <td>-0.127569</td>
    </tr>
    <tr>
      <th>sum_artist_popularity</th>
      <td>0.275534</td>
      <td>0.057902</td>
      <td>0.000024</td>
      <td>-0.010588</td>
      <td>-0.028688</td>
      <td>0.090585</td>
      <td>0.001825</td>
      <td>0.050047</td>
      <td>0.030925</td>
      <td>-0.093217</td>
      <td>-0.041702</td>
      <td>-0.011954</td>
      <td>0.227238</td>
      <td>0.508247</td>
      <td>0.373577</td>
      <td>1.000000</td>
      <td>-0.109790</td>
    </tr>
    <tr>
      <th>age</th>
      <td>-0.557065</td>
      <td>-0.039230</td>
      <td>-0.237185</td>
      <td>-0.409324</td>
      <td>-0.449045</td>
      <td>0.036217</td>
      <td>0.447438</td>
      <td>0.178743</td>
      <td>0.023298</td>
      <td>0.032377</td>
      <td>-0.098451</td>
      <td>-0.104819</td>
      <td>-0.110662</td>
      <td>-0.181174</td>
      <td>-0.127569</td>
      <td>-0.109790</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


