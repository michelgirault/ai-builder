#!/bin/bash
#custom script for ai model workflow builder

if [ -d $LANGFLOW_DATASETS_DIR ]; then
    echo "Directory exists."
else
    mkdir $LANGFLOW_DATASETS_DIR 
fi


aws s3 sync --endpoint-url=$S3_ENDPOINT $LANGFLOW_DATASETS_DIR $S3_BUCKET

chroma run --host localhost --port 8000 --path /app/chrome &
python3 -m langflow run 
wait