#!/bin/bash
#custom script for ai model workflow builder
aws configure set aws_access_key_id $aws_access_key_id
aws configure set aws_secret_access_key $aws_secret_access_key

chroma run --host localhost --port 3333 --path /data/chroma &
flowise start
wait