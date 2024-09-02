#!/bin/bash
#custom script for ai model workflow builder

if [ -d $LANGFLOW_DATASETS_DIR ]; then
    echo "Directory exists."
else
    mkdir $LANGFLOW_DATASETS_DIR 
fi
#sync the drive from s3
aws s3 sync --endpoint-url=$S3_ENDPOINT $LANGFLOW_DATASETS_DIR $S3_BUCKET
#run the vector db

chroma run --host localhost --port 8000 --path /app/chrome &
#run the ai builder
python3 -m langflow run 
wait