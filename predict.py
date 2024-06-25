import os
import time
import torch
import warnings
import subprocess
from threading import Thread
from cog import BasePredictor, Input, ConcatenateIterator
import transformers
from transformers.generation import GenerationConfig, TextIteratorStreamer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
)

# Configuration
USE_PREDOWNLOADED_WEIGHTS = True  # Set to False to download from Hugging Face
MODEL_FAMILY = "1.5B"  # Choose from "0.5B", "1.5B", "7B", "57B", "72B"

MODEL_VARIANTS = {
    "72B": [
        "Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen2-72B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2-72B-Instruct-GPTQ-Int4",
    ],
    "57B": [
        "Qwen/Qwen2-57B-A14B-Instruct",
        "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4",
    ],
    "7B": [
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
    ],
    "1.5B": [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
    ],
    "0.5B": [
        "Qwen/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
    ],
}

MODEL_NAMES = {f"{name.split('/')[-1]}": name for name in MODEL_VARIANTS[MODEL_FAMILY]}
DEFAULT_MODEL = list(MODEL_NAMES.keys())[0]  # First model in the list

MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

BASE_URL = f"https://weights.replicate.delivery/default/Qwen2-{MODEL_FAMILY}-Instruct/"

# Environment setup
if USE_PREDOWNLOADED_WEIGHTS:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def get_model_info(model_type):
    model_name = MODEL_NAMES[model_type].split("/")[-1]
    model_dir = model_name
    if "GPTQ" in model_name:
        # For GPTQ models, we need to include the quantization info in the directory name
        quantization_info = "-".join(model_name.split("-")[-2:])
        model_dir = f"Qwen2-{MODEL_FAMILY}-Instruct-{quantization_info}"
    else:
        model_dir = f"Qwen2-{MODEL_FAMILY}-Instruct"
    return {
        "dir": model_dir,
        "file": f"models--Qwen--{model_name}.tar",
    }


def get_quantization_config(model_name):
    if "GPTQ" in model_name:
        bits = 4 if "Int4" in model_name else 8
        print(f"[!] Using GPTQ quantization with {bits} bits")
        return GPTQConfig(bits=bits, disable_exllama=True)
    elif "AWQ" in model_name:
        print("[!] Using AWQ quantization with 4 bits")
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    else:
        print("[!] Using no quantization")
        return None


def download_weights(url: str, dest: str) -> None:
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
        print(f"[!] Setting up default model: {MODEL_NAMES[DEFAULT_MODEL]}")

        warnings.filterwarnings("ignore")
        transformers.logging.set_verbosity_error()

        self.models = {}
        self.tokenizers = {}

        # Download and set up the default model
        self.load_model(DEFAULT_MODEL)

    def load_model(self, model_type):
        if model_type not in self.models:
            model_name = MODEL_NAMES[model_type]
            print(f"[!] Loading model: {model_name}")

            if USE_PREDOWNLOADED_WEIGHTS:
                # Code for using pre-downloaded weights
                model_info = get_model_info(model_type)
                model_dir = model_info["dir"]
                file_name = model_info["file"]

                model_url = f"{BASE_URL}{model_dir}/model-cache/{file_name}"
                token_url = f"{BASE_URL}{model_dir}/token-cache/{file_name}"

                # Lazy downloading
                model_path = os.path.join(MODEL_CACHE, file_name)
                if not os.path.exists(model_path.replace(".tar", "")):
                    download_weights(model_url, model_path)

                token_path = os.path.join(TOKEN_CACHE, file_name)
                if not os.path.exists(token_path.replace(".tar", "")):
                    download_weights(token_url, token_path)

            # Common code for both pre-downloaded and Hugging Face
            self.tokenizers[model_type] = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, cache_dir=TOKEN_CACHE
            )

            quantization_config = get_quantization_config(model_name)
            model_kwargs = {
                "trust_remote_code": True,
                "cache_dir": MODEL_CACHE,
                "device_map": "auto",
            }
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            model.generation_config = GenerationConfig.from_pretrained(
                model_name, trust_remote_code=True, cache_dir=MODEL_CACHE
            )
            self.models[model_type] = model

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Give me a short introduction to large language model.",
        ),
        system_prompt: str = Input(
            description="System prompt", default="You are a helpful assistant."
        ),
        model_type: str = Input(
            description=f"Choose from available {MODEL_FAMILY} models",
            default=DEFAULT_MODEL,
            choices=list(MODEL_NAMES.keys()),
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
        if model_type not in self.models:
            self.load_model(model_type)

        model = self.models[model_type]
        tokenizer = self.tokenizers[model_type]

        if seed is None:
            seed = torch.randint(0, 2**30, (1,)).item()
        torch.manual_seed(seed)
        print("Using seed:", seed)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
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

        start_time = time.time()
        first_token_time = None
        total_tokens = 0
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            total_tokens += len(new_text)
            yield new_text
        thread.join()
        end_time = time.time()

        total_time = end_time - start_time
        throughput = total_tokens / total_time if total_time > 0 else 0

        print(f"\nTime to first token: {first_token_time:.2f} seconds")
        print(f"Total generation time: {total_time:.2f} seconds")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Throughput: {throughput:.2f} tokens/second")
