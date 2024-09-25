import io
import os
import joblib
import subprocess

import boto3

# Download Model
import json
import sagemaker
from sagemaker import get_execution_role
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

sagemaker_session = sagemaker.Session()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

bucket_name = sagemaker_session.default_bucket()
job_prefix = f"train-{model_id.split('/')[-1].replace('.', '-')}"

def get_last_job_name(job_name_prefix):
    import boto3
    sagemaker_client = boto3.client('sagemaker')
    
    search_response = sagemaker_client.search(
        Resource='TrainingJob',
        SearchExpression={
            'Filters': [
                {
                    'Name': 'TrainingJobName',
                    'Operator': 'Contains',
                    'Value': job_name_prefix
                },
                {
                    'Name': 'TrainingJobStatus',
                    'Operator': 'Equals',
                    'Value': "Completed"
                }
            ]
        },
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1)
    
    return search_response['Results'][0]['TrainingJob']['TrainingJobName']

job_name = get_last_job_name(job_prefix)

# Inference configurations
instance_count = 1
instance_type = "ml.g5.8xlarge"
number_of_gpu = 1
health_check_timeout = 700

image_uri = get_huggingface_llm_image_uri(
    "huggingface",
    version="1.4"
)

model = HuggingFaceModel(
    image_uri=image_uri,
    model_data=f"s3://{bucket_name}/{job_name}/{job_name}/output/model.tar.gz",
    role=get_execution_role(),
    env={
        'HF_MODEL_ID': "/opt/ml/model", # path to where sagemaker stores the model
        'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
        'HF_MODEL_QUANTIZE': "bitsandbytes"
    }
)

predictor = model.deploy(
    initial_instance_count=instance_count,
    instance_type=instance_type,
    container_startup_health_check_timeout=health_check_timeout,
    model_data_download_timeout=3600
)

# Predict
from sagemaker.huggingface.model import HuggingFacePredictor

endpoint_name = "<ENDPOINT_NAME>" #Required if you want to create a predictor without running the previous code

if 'predictor' not in locals() and 'predictor' not in globals():
    print("Create predictor")
    predictor = HuggingFacePredictor(
        endpoint_name=endpoint_name
    )

base_prompt = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {{question}}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
"""

prompt = base_prompt.format(question="What is the context window of Anthropic Claude 2.1 model?")

predictor.predict({
	"inputs": prompt,
    "parameters": {
        "n_predict": -1,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["<|start_header_id|>", "<|eot_id|>", "<|start_header_id|>user<|end_header_id|>", "assistant"]
    }
})


