"""Connect to Databricks cluster and read data from a demo table (for testing purposes)."""

from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.profile("dbc-643c4c2b-d6c9").getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)
