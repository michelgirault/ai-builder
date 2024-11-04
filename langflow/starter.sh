#!/bin/bash
#custom script for ai model workflow builder

aws configure set aws_access_key_id $aws_access_key_id
aws configure set aws_secret_access_key $aws_secret_access_key

if [ -d $LANGFLOW_DATASETS_DIR ]; then
    echo "Directory exists."
else
    mkdir $LANGFLOW_DATASETS_DIR 
fi
#sync the drive from s3 
aws s3 sync --endpoint-url=$S3_ENDPOINT $S3_BUCKET $LANGFLOW_DATASETS_DIR 
#run the vector db

chroma run --host 0.0.0.0 --port 8000 --path /app/chrome &
#run the ai builder
python3 -m langflow run 
wait