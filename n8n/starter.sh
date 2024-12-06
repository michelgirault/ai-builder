#!/bin/bash
#custom script for ai model workflow builder
#make sure the folder existing within the volume

#install huggingface cli
pip install -U "huggingface_hub[cli]"

#config git warning messages
git config --global --add safe.directory /app

#start huggingface login
huggingface-cli login --token ${HF_TOKEN} --add-to-git-credential

#start n8n server
if [ "$#" -gt 0 ]; then
  # Got started with arguments
  exec n8n "$@"
  
else
  # Got started without arguments
  exec n8n
fi