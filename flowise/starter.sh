#!/bin/bash
#custom script for ai model workflow builder

chroma run --host localhost --port 8000 --path ./my_chroma_data &
flowise start
wait