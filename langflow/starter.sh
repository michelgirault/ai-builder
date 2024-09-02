#!/bin/bash
#custom script for ai model workflow builder

chroma run --host localhost --port 8000 --path /app/chrome &
python3 -m langflow run 
wait