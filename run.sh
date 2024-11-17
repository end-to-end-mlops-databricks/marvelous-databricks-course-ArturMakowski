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

# Run the function
deploy_package
