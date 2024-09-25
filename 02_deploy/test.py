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