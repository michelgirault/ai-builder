#!/bin/bash
set -a
source .env
set +a

s3cmd sync --exclude '.git/*' ${MODEL_LOCAL_FOLDER} ${S3_MODEL_BASE} --host=${S3_HOST} --host-bucket=${S3_BUCKET} --secret_key=${S3_SECRET_KEY} --access_key=${S3_ACCESS_KEY}