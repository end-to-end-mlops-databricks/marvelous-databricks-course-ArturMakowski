#!/bin/bash

function deploy_package() {
    echo "Building package with uv build..."
    uv build

    if [ $? -eq 0 ]; then
        echo "Building successful, copying to Databricks filesystem..."
        databricks fs cp --overwrite --profile=DEFAULT \
            dist/mlops_with_databricks-0.0.1-py3-none-any.whl \
            dbfs:/Volumes/mlops_students/armak58/packages/mlops_with_databricks-0.0.1-py3-none-any.whl

        if [ $? -eq 0 ]; then
            echo "Package successfully deployed to Databricks"
        else
            echo "Error: Failed to copy package to Databricks"
            return 1
        fi
    else
        echo "Error: Build failed"
        return 1
    fi
}

function run_databricks_bundle_stage() {
    echo "Running Databricks bundle deploy..."
    databricks bundle deploy --profile DEFAULT --target stage
    echo "Databricks bundle deploy successful"
    echo "Generating data..."
    "/Users/arturmakowski/Documents/Python_projects/marvelous-databricks-course-ArturMakowski/.venv/bin/python" "/Users/arturmakowski/.vscode/extensions/databricks.databricks-2.4.8-darwin-arm64/resources/python/dbconnect-bootstrap.py" "/Users/arturmakowski/Documents/Python_projects/marvelous-databricks-course-ArturMakowski/src/mlops_with_databricks/pipeline/generate_data.py"
    echo "Data generated successfully"
    echo "Running Databricks bundle run..."
    databricks bundle run --profile DEFAULT --target stage
    echo "Databricks bundle run successful"
}

function run_databricks_bundle_prod() {
    echo "Running Databricks bundle deploy..."
    databricks bundle deploy --profile DEFAULT --target prod
    echo "Databricks bundle deploy successful"
    echo "Generating data..."
    "/Users/arturmakowski/Documents/Python_projects/marvelous-databricks-course-ArturMakowski/.venv/bin/python" "/Users/arturmakowski/.vscode/extensions/databricks.databricks-2.4.8-darwin-arm64/resources/python/dbconnect-bootstrap.py" "/Users/arturmakowski/Documents/Python_projects/marvelous-databricks-course-ArturMakowski/src/mlops_with_databricks/pipeline/generate_data.py"
    echo "Data generated successfully"
    echo "Running Databricks bundle run..."
    databricks bundle run --profile DEFAULT --target prod
    echo "Databricks bundle run successful"
}
