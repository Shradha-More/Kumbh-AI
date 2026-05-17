"""
QLoRA fine-tuning script for Qwen2.5-7B-Instruct on Kumbh Mela 2027 data.
Uses Unsloth for fast training (2x faster, 60% less VRAM).

Requirements:
  pip install unsloth transformers datasets peft trl bitsandbytes accelerate

Hardware: 16GB+ VRAM GPU recommended (works on RTX 3090/4090, A100)
          8GB VRAM: switch to Qwen2.5-3B-Instruct
          No GPU: NOT recommended — use cloud (RunPod/Vast.ai/Colab Pro)

Usage:
  python train_qlora.py --model qwen2.5-7b --data ../data/synthetic_qa/all_languages_combined.jsonl
"""

import os
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

import torch
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

MODEL_OPTIONS = {
    "qwen2.5-7b":  "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-3b":  "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi4-mini":   "microsoft/Phi-4-mini-instruct",
}

SYSTEM_PROMPTS = {
    "en": "You are a helpful multilingual assistant for Nashik Kumbh Mela 2027. Answer questions about schedules, ghats, temples, transport, emergency services, and tourist places in Nashik. Be accurate, concise, and helpful. Only answer based on the provided context.",
    "hi": "आप नाशिक कुंभ मेला 2027 के लिए एक सहायक बहुभाषी सहायक हैं। कार्यक्रम, घाट, मंदिर, परिवहन, आपातकालीन सेवाओं और नाशिक में पर्यटन स्थलों के बारे में प्रश्नों का उत्तर दें।",
    "mr": "तुम्ही नाशिक कुंभ मेळा 2027 साठी एक सहाय्यक बहुभाषी सहाय्यक आहात. वेळापत्रक, घाट, मंदिरे, वाहतूक, आणीबाणी सेवा आणि नाशिकमधील पर्यटन स्थळांबद्दल प्रश्नांची उत्तरे द्या.",
    "gu": "તમે નાશિક કુંભ મેળા 2027 માટે એક સહાયક બહુભાષી સહાયક છો. સમયપત્રક, ઘાટ, મંદિરો, પરિવહન, કટોકટી સેવાઓ અને નાશિકમાં પ્રવાસી સ્થળો વિશે પ્રશ્નોના જવાબ આપો.",
    "ta": "நீங்கள் நாசிக் கும்பமேளா 2027-க்கான உதவிகரமான பன்மொழி உதவியாளர். அட்டவணைகள், கடவுகள், கோயில்கள், போக்குவரத்து, அவசர சேவைகள் மற்றும் நாசிக்கில் சுற்றுலா இடங்கள் பற்றிய கேள்விகளுக்கு பதிலளிக்கவும்.",
    "te": "మీరు నాసిక్ కుంభమేళా 2027 కోసం సహాయకరమైన బహుభాషా సహాయకులు. షెడ్యూల్‌లు, ఘాట్‌లు, దేవాలయాలు, రవాణా, అత్యవసర సేవలు మరియు నాసిక్‌లో పర్యాటక ప్రదేశాల గురించి ప్రశ్నలకు సమాధానం ఇవ్వండి.",
    "kn": "ನೀವು ನಾಸಿಕ್ ಕುಂಭಮೇಳ 2027 ಗಾಗಿ ಸಹಾಯಕ ಬಹುಭಾಷಾ ಸಹಾಯಕರು. ವೇಳಾಪಟ್ಟಿಗಳು, ಘಾಟ್‌ಗಳು, ದೇವಸ್ಥಾನಗಳು, ಸಾರಿಗೆ, ತುರ್ತು ಸೇವೆಗಳು ಮತ್ತು ನಾಸಿಕ್‌ನಲ್ಲಿ ಪ್ರವಾಸಿ ತಾಣಗಳ ಬಗ್ಗೆ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಿ.",
    "ml": "നിങ്ങൾ നാസിക് കുംഭമേള 2027-നുള്ള സഹായകരമായ ബഹുഭാഷ സഹായിയാണ്. ഷെഡ്യൂളുകൾ, ഘട്ടുകൾ, ക്ഷേത്രങ്ങൾ, ഗതാഗതം, അടിയന്തര സേവനങ്ങൾ, നാസിക്കിലെ ടൂറിസ്റ്റ് സ്ഥലങ്ങൾ എന്നിവയെക്കുറിച്ചുള്ള ചോദ്യങ്ങൾക്ക് ഉത്തരം നൽകൂ.",
}


def load_training_data(data_path: str, max_samples: int = None) -> Dataset:
    """Load JSONL training data and format as chat messages."""
    log.info(f"Loading data from: {data_path}")
    path = Path(data_path)

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    log.info(f"Loaded {len(records)} records")

    if max_samples:
        import random
        random.shuffle(records)
        records = records[:max_samples]
        log.info(f"Sampling {max_samples} records")

    # Format as chat messages for SFT
    formatted = []
    for r in records:
        lang = r.get("language", "en")
        system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
        instruction = r.get("instruction", "")
        context = r.get("input", "")
        output = r.get("output", "")

        if not instruction or not output:
            continue

        user_msg = instruction
        if context:
            user_msg = f"Context: {context}\n\nQuestion: {instruction}"

        formatted.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ]
        })

    log.info(f"Formatted {len(formatted)} training examples")
    return Dataset.from_list(formatted)


def train_with_unsloth(model_name: str, dataset: Dataset, output_dir: str,
                       max_steps: int = 1000, learning_rate: float = 2e-4):
    """Fine-tune using Unsloth for fast QLoRA training."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Install unsloth: pip install unsloth")

    log.info(f"Loading model with Unsloth: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,  # Auto detect
        load_in_4bit=True,
    )

    log.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Format dataset using chat template
    def format_chat(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    formatted_dataset = dataset.map(format_chat, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=output_dir,
            save_steps=200,
            save_total_limit=3,
            report_to="none",
        ),
    )

    log.info("Starting training...")
    trainer.train()

    log.info(f"Saving model to {output_dir}/final")
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    return model, tokenizer


def train_standard_qlora(model_name: str, dataset: Dataset, output_dir: str,
                         max_steps: int = 1000, learning_rate: float = 2e-4):
    """Fall back to standard HuggingFace QLoRA if Unsloth not available."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    log.info(f"Loading model (standard QLoRA): {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def format_chat(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    formatted_dataset = dataset.map(format_chat, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=50,
            max_steps=max_steps,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            optim="paged_adamw_8bit",
            output_dir=output_dir,
            save_steps=200,
            save_total_limit=3,
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    return model, tokenizer


def export_to_gguf(model_dir: str, output_dir: str, quantization: str = "q4_k_m"):
    """Export the fine-tuned model to GGUF format for llama.cpp inference."""
    log.info(f"Exporting to GGUF ({quantization})...")
    log.info("This requires llama.cpp installed. Run the following commands:")
    print(f"""
# 1. Convert to GGUF fp16 first
python llama.cpp/convert_hf_to_gguf.py {model_dir}/final \\
    --outfile {output_dir}/kumbh_model_fp16.gguf \\
    --outtype f16

# 2. Quantize to Q4_K_M (recommended — 4GB, fast, good quality)
./llama.cpp/llama-quantize {output_dir}/kumbh_model_fp16.gguf \\
    {output_dir}/kumbh_model_{quantization}.gguf {quantization.upper()}

# 3. Test with llama.cpp
./llama.cpp/llama-cli -m {output_dir}/kumbh_model_{quantization}.gguf \\
    -p "रामकुंड कहां है?" --temp 0.3
    """)


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for Kumbh Mela AI")
    parser.add_argument("--model", default="qwen2.5-7b", choices=MODEL_OPTIONS.keys())
    parser.add_argument("--data", default=str(ROOT / "data/synthetic_qa/all_languages_combined.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "models/kumbh_model"))
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use-unsloth", action="store_true", default=True)
    parser.add_argument("--export-gguf", action="store_true")
    args = parser.parse_args()

    model_id = MODEL_OPTIONS[args.model]
    log.info(f"Model: {model_id}")
    log.info(f"Data: {args.data}")
    log.info(f"Output: {args.output}")
    log.info(f"Max steps: {args.max_steps}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    dataset = load_training_data(args.data, args.max_samples)

    # Split 95/5 train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_data = split["train"]
    log.info(f"Training samples: {len(train_data)}")

    if args.use_unsloth:
        try:
            model, tokenizer = train_with_unsloth(
                model_id, train_data, args.output, args.max_steps, args.lr
            )
        except ImportError:
            log.warning("Unsloth not found, falling back to standard QLoRA")
            model, tokenizer = train_standard_qlora(
                model_id, train_data, args.output, args.max_steps, args.lr
            )
    else:
        model, tokenizer = train_standard_qlora(
            model_id, train_data, args.output, args.max_steps, args.lr
        )

    log.info("Training complete!")
    log.info(f"Model saved to: {args.output}/final")

    if args.export_gguf:
        export_to_gguf(args.output, args.output)


if __name__ == "__main__":
    main()
