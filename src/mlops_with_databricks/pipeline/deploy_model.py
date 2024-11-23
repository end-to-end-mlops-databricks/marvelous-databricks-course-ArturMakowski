"""This script is used to deploy the model to the serving endpoint. The model version is fetched from the evaluate_model task."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput

from mlops_with_databricks.data_preprocessing.dataclasses import DatabricksConfig, ModelConfig, ModelServingConfig

workspace = WorkspaceClient()


model_version = workspace.dbutils.jobs.taskValues.get(taskKey="evaluate_model", key="model_version")


catalog_name = DatabricksConfig.catalog_name
schema_name = DatabricksConfig.schema_name

workspace.serving_endpoints.update_config_and_wait(
    name=ModelServingConfig.serving_endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=f"{catalog_name}.{schema_name}.{ModelConfig.model_name}",
            scale_to_zero_enabled=True,
            workload_size="Small",
            entity_version=model_version,
        )
    ],
)
