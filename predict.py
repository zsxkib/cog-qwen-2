# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
import torch
import warnings
import subprocess
from threading import Thread
from cog import BasePredictor, Input, ConcatenateIterator
from transformers.generation import GenerationConfig, TextIteratorStreamer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    GPTQConfig,
)

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

BASE_URL = f"https://weights.replicate.delivery/default/Qwen2-0.5B-Instruct-GPTQ-Int4/{MODEL_CACHE}/"

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def get_quantization_config(model_name):
    if "GPTQ" in model_name:
        bits = 4 if "Int4" in model_name else 8
        return GPTQConfig(bits=bits, disable_exllama=True)
    elif "AWQ" in model_name:
        # AWQ typically uses 4-bit quantization
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    elif "GGUF" in model_name:
        # GGUF models are typically loaded differently, often using libraries like ctransformers
        raise NotImplementedError("GGUF models are not supported in this setup")
    else:
        # For non-quantized models, return None
        return None


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Add these lines at the beginning of the setup method
        import warnings
        import transformers
        
        # Ignore all warnings
        warnings.filterwarnings("ignore")
        
        # Disable transformers warnings
        transformers.logging.set_verbosity_error()

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        model_files = [
            "models--Qwen--Qwen2-0.5B-Instruct-GPTQ-Int4.tar",
        ]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        if not os.path.exists(TOKEN_CACHE):
            os.makedirs(TOKEN_CACHE)

        token_files = [
            "models--Qwen--Qwen2-0.5B-Instruct-GPTQ-Int4.tar",
        ]
        for token_file in token_files:
            url = BASE_URL + token_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(TOKEN_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, cache_dir=TOKEN_CACHE
        )

        # Get the appropriate quantization config
        quantization_config = get_quantization_config(MODEL_NAME)

        # Load the model
        model_kwargs = {
            "trust_remote_code": True,
            "cache_dir": MODEL_CACHE,
            "device_map": "auto",
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

        model.generation_config = GenerationConfig.from_pretrained(
            MODEL_NAME, trust_remote_code=True, cache_dir=MODEL_CACHE
        )
        self.model = model

        print(f"Model device map: {self.model.hf_device_map}")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Give me a short introduction to large language model.",
        ),
        system_prompt: str = Input(
            description="System prompt", default="You are a helpful assistant."
        ),
        max_new_tokens: int = Input(
            description="The maximum number of tokens to generate",
            default=512,
            ge=1,
            le=32768,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            default=1.0,
            ge=0.1,
            le=5.0,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens.",
            default=1.0,
            ge=0.01,
            le=1.0,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens.",
            default=1,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            default=1.0,
            ge=0.01,
            le=10.0,
        ),
        seed: int = Input(
            description="The seed for the random number generator", default=None
        ),
    ) -> ConcatenateIterator:
        """Run a single prediction on the model"""
        if seed is None:
            seed = torch.randint(0, 2**30, (1,)).item()
        torch.manual_seed(seed)
        print("Using seed:", seed)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = {
            "input_ids": model_inputs.input_ids,
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "streamer": streamer,
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text
        thread.join()
