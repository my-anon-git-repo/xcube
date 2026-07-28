import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from pathlib import Path
from datetime import datetime
import textwrap

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def smart_load_model(
    model_name: str,
    max_vram_gb: float = 80.0,          # Your GPU VRAM in GB
    safety_margin: float = 0.75,         # Use only 75% of VRAM to leave headroom for KV cache/overhead
    force_quant: bool = False,           # Override to always quantize (useful for testing)
    flash_attn: bool = True,             # Use flash attention if installed
    trust_remote_code: bool = True,
    **from_pretrained_kwargs
):
    """
    Smartly loads a model:
    - Parses model_name for size (e.g. '80B', '30B-A3B', '235B')
    - Estimates VRAM for full precision (~2 GB / B params)
    - If too big (> ~75% of max_vram_gb), uses 4-bit quantization
    - Falls back to bfloat16 for better numerical stability
    """
    # Extract approximate total params in billions from model name
    size_match = re.search(r'(\d+\.?\d*)\s*(B|b)', model_name, re.IGNORECASE)
    if size_match:
        param_billions = float(size_match.group(1))
    else:
        # Fallback: assume small if no match (or you can raise error)
        param_billions = 7.0
        print("Warning: Could not parse param size from model_name. Assuming small (~7B).")

    # MoE note: we still use TOTAL params for VRAM estimate (all weights loaded)
    estimated_vram_full_gb = param_billions * 2.0 * 1.25  # 2 bytes/param + ~25% overhead

    available_safe_gb = max_vram_gb * safety_margin
    should_quantize = force_quant or (estimated_vram_full_gb > available_safe_gb)

    print(f"Model: {model_name}")
    print(f"Estimated params: ~{param_billions:.1f}B")
    print(f"Estimated VRAM (bf16/full): ~{estimated_vram_full_gb:.1f} GB")
    print(f"Available safe VRAM: ~{available_safe_gb:.1f} GB → Quantize? {should_quantize}")

    quantization_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if should_quantize:
        print("→ Applying 4-bit quantization (NF4 + double quant)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",              # Best for reasoning models
            bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
            bnb_4bit_use_double_quant=True,         # Extra compression
        )
        # When quantized, we can often afford bf16 compute even on tight VRAM
    else:
        print("→ Loading in full precision (bf16)")

    attn_impl = "flash_attention_2" if flash_attn else "eager"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            **from_pretrained_kwargs
        )
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Loading failed: {e}")
        print("Suggestions:")
        print("- Try a pre-quantized variant (e.g. -AWQ, -GPTQ, -FP8 suffix)")
        print("- Lower safety_margin or set force_quant=True")
        print("- Use vLLM / llama.cpp for better MoE / long-context handling")
        raise


# ────────────────────────────────────────────────
# Example usage – replace with your model
# ────────────────────────────────────────────────

# model_name = "Qwen/Qwen3-Next-80B-A3B-Thinking"          # ~80B total → should quantize
# # model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507"         # ~30B → likely full precision
# # model_name = "meta-llama/Llama-3.1-8B-Instruct"         # small → full

# model = smart_load_model(
#     model_name,
#     max_vram_gb=80.0,
#     safety_margin=0.80,          # Tune: 0.7–0.9 depending on how much KV cache you need
#     force_quant=False,           # Set True to always quantize
#     flash_attn=True,             # pip install flash-attn --no-build-isolation first
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def classify_snippet(
    PROMPT,
    model_name="microsoft/Phi-3-mini-4k-instruct",
    output_dir="llm_outputs",
    filename_prefix="reasoning",
    # You can pass any generate() kwargs you want here
    **generate_kwargs
):
    """
    Generate with live console streaming + save full output + actual generation params to file.
    All extra kwargs passed to this function are forwarded to model.generate().
    """
    # ────────────────────────────────────────────────
    # Default generation parameters (only used if not overridden)
    default_generate_kwargs = {
        "max_new_tokens": 2048,
        "do_sample": False,
        "temperature": 0.0,
        "repetition_penalty": 1.12,
    }

    # Merge defaults with user-provided kwargs (user wins)
    gen_params = {**default_generate_kwargs, **generate_kwargs}

    # Fix: If do_sample is False, ensure temperature and top_p are not set
    # if not gen_params.get("do_sample", False):
    #     # Remove sampling-specific parameters when using greedy decoding
    #     if "temperature" in gen_params:
    #         gen_params["temperature"] = None
    #     if "top_p" in gen_params:
    #         gen_params["top_p"] = None
    #     if "top_k" in gen_params:
    #         gen_params["top_k"] = None

    # ────────────────────────────────────────────────
    # Prepare output file
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"{filename_prefix}_{timestamp}.txt"
    print(f"Will save full output to: {output_file.absolute()}")
    print("-" * 70)

    # ────────────────────────────────────────────────
    # Load tokenizer & model
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,   # important for Qwen chat template
    )

    # model = smart_load_model(
    #     model_name,
    #     max_vram_gb=80.0,
    #     safety_margin=0.05,          # Tune: 0.7–0.9 depending on how much KV cache you need
    #     force_quant=False,           # Set True to always quantize
    #     flash_attn=True,             # pip install flash-attn --no-build-isolation first
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # FP8 weights are loaded, compute in bf16
        device_map="auto",
        # attn_implementation="flash_attention_2",  # Strongly recommended for memory/speed
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )

    # Set padding token if not set (important for DeepSeek models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare prompt
    # DeepSeek chat models expect a specific chat format
    if "chat" in model_name.lower():
        messages = [
            {"role": "user", "content": PROMPT}
        ]

        # messages = [
        #     {"role": "system", "content": f"You are a strict multi-label classifier. Only use categories from the provided list. Do not invent new categories."},
        #     {"role": "user", "content": f"Valid categories:\n{category_list_str}\n\nSnippet:\n{snippet_text}\n\nClassify this snippet using ONLY the valid categories above."}
        # ]

        # Apply chat template and get input IDs
        input_text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
    else:
        messages = [
            {"role": "system", "content": "You are a strict multi-label classifier."},
            {"role": "user", "content": PROMPT},
        ]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)

        # inputs = tokenizer(PROMPT, return_tensors="pt", padding=True).to(model.device)

    # ────────────────────────────────────────────────
    # Streamer for live console output
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    print("\nGenerating (reasoning tokens appear live below)...\n")

    # ────────────────────────────────────────────────
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            streamer=streamer,
            return_dict_in_generate=True,
            output_scores=False,
            **gen_params   # ← all params come from here
        )

    # ────────────────────────────────────────────────
    # Extract generated text
    generated_ids = output.sequences[0]
    prompt_length = inputs.input_ids.shape[1]
    new_ids = generated_ids[prompt_length:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # ────────────────────────────────────────────────
    # Format the actual generation parameters nicely for logging
    param_lines = []
    for k, v in gen_params.items():
        param_lines.append(f"  {k: <18}: {v}")

    # ────────────────────────────────────────────────
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write("Generation parameters:\n")
        f.write("\n".join(param_lines) + "\n")
        f.write("=" * 70 + "\n\n")
        f.write("=== PROMPT ===\n")
        f.write(PROMPT.strip() + "\n\n")
        f.write("=== GENERATED OUTPUT ===\n")

        if generated_text.strip():
            wrapped = textwrap.fill(
                generated_text,
                width=88,
                break_long_words=False,
                replace_whitespace=False
            )
            f.write(wrapped + "\n\n")
        else:
            f.write("[No new tokens generated]\n")

        f.write("\n" + "=" * 70 + "\n")

    # ────────────────────────────────────────────────
    # Final feedback
    if generated_text.strip():
        print("\n" + "=" * 70)
        print("Generation finished. Full output saved to:")
        print(output_file)
    else:
        print("\n" + "=" * 70)
        print("Warning: Model generated NO new tokens.")
        print("Full log saved anyway to:")
        print(output_file)

    return generated_text

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    import sys
    from pathlib import Path
    sys.path.append(str(Path.home() / ".xcube/data/LF-WikiSeeAlso-320K_sample"))
    from prompt_test import PROMPT

    # good (but not thinking model, directly answer)
    # result = classify_snippet(
    #     PROMPT,
    #     model_name="Qwen/Qwen2.5-14B-Instruct", 
    #     output_dir="llm_outputs",
    #     filename_prefix="reasoning",
    #     max_new_tokens = 32768,
    #     do_sample = False,
    #     temperature = 0.0,
    #     top_p = 1.0,
    #     repetition_penalty=1.05,
    # )

    # good 
    result = classify_snippet(
        PROMPT,
        model_name="Qwen/Qwen3-30B-A3B-Thinking-2507", 
        output_dir="llm_outputs",
        filename_prefix="reasoning",
        max_new_tokens = 32768,
        do_sample = True,
        temperature = 0.2,
        top_p = 1.0,
        repetition_penalty=1.05,
    )


    # good
    # result = classify_snippet(
    #     PROMPT,
    #     model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
    #     output_dir="llm_outputs",
    #     filename_prefix="reasoning",
    #     max_new_tokens = 32768,
    #     do_sample = False,
    #     temperature = 0.0,
    #     top_p = 1.0,
    #     repetition_penalty=1.05,
    # )

    # good
    # result = classify_snippet(
    #     PROMPT,
    #     model_name="openai/gpt-oss-20b", 
    #     output_dir="llm_outputs",
    #     filename_prefix="reasoning",
    #     max_new_tokens = 32768,
    #     do_sample = False,
    #     temperature = 0.0,
    #     top_p = 1.0,
    #     repetition_penalty=1.05,
    # )

    # good
    # result = classify_snippet(
    #     PROMPT,
    #     model_name="deepseek-ai/deepseek-llm-7b-chat",
    #     output_dir="llm_outputs",
    #     filename_prefix="reasoning",
    #     max_new_tokens=32768,
    #     do_sample=True,
    #     temperature=0.1,
    #     top_p=0.9,
    #     repetition_penalty=1.12,
    # )

    # good
    # result = classify_snippet(
    #     PROMPT,
    #     model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    #     output_dir="llm_outputs",
    #     filename_prefix="reasoning",
    #     max_new_tokens=32768,
    #     do_sample=False,          
    #     temperature=0.0,         
    #     top_p=0.95,              
    #     repetition_penalty=1.1,  
    # )

    # good
    # result = classify_snippet(
    #     PROMPT,
    #     model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    #     output_dir="llm_outputs",
    #     filename_prefix="reasoning",
    #     max_new_tokens = 32768,
    #     do_sample = False,
    #     temperature = 0.0,
    #     top_p=0.95,
    #     repetition_penalty = 1.12,
    # )
