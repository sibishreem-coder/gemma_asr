import unsloth
import os
import gc
import random
import torch
import wandb
import datasets
import soundfile as sf
import numpy as np
import jiwer

from dotenv import load_dotenv
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from datasets import load_dataset, Audio
from transformers import TrainerCallback
from unsloth import FastModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
WB_TOKEN = os.environ.get("WB_API_KEY", "")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH     = "unsloth/gemma-4-E4B-it"
RUN_NAME       = "gemma4-june"
HUB_MODEL_ID   = "Sibishreekapture/gemma4-asr-tamil-j1"
MAX_SEQ_LENGTH = 2048
TARGET_SR      = 16000

TRAIN_SAMPLES  = {"tamil": 16000, "English": 2500}
EVAL_SAMPLES   = {"tamil": 30,  "English": 10}

FLEURS_CONFIGS = [("tamil", "tamil", "ta")]
SVARAH_CONFIGS = [("English", "default", "svarah_en")]

DISK_ROOT       = "./asr_data_fleurs"
TRAIN_AUDIO_DIR = os.path.join(DISK_ROOT, "train_wavs")
EVAL_AUDIO_DIR  = os.path.join(DISK_ROOT, "eval_wavs")

LORA_R          = 8
LORA_ALPHA      = 16
EPOCHS          = 1
BATCH_SIZE      = 4
GRAD_ACCUM      = 4
LEARNING_RATE   = 1e-4


# ── Callbacks ─────────────────────────────────────────────────────────────────
class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step      = state.global_step
            loss      = logs.get("loss", "")
            eval_loss = logs.get("eval_loss", "")
            if loss:      print(f"Step {step} | train_loss: {loss:.4f}")
            if eval_loss: print(f"Step {step} | eval_loss:  {eval_loss:.4f}")


class WandbMetricsCallback(TrainerCallback):
    def __init__(self, processor, eval_entries, sample_size=10):
        self.processor    = processor
        self.eval_entries = eval_entries
        self.sample_size  = sample_size

    def compute_wer(self, model):
        model.eval()
        preds   = {}
        refs    = {}
        skipped = 0

        indices = random.sample(
            range(len(self.eval_entries)),
            min(len(self.eval_entries), self.sample_size)
        )

        for idx in indices:
            entry = self.eval_entries[idx]
            lang  = entry.get("lang", "Tamil")

            try:
                audio_array, _ = sf.read(entry["audio"], dtype="float32")
            except Exception as e:
                print(f"Audio read error: {e}")
                skipped += 1
                continue

            duration = len(audio_array) / TARGET_SR
            if duration > 20.0 or duration < 1.0:
                skipped += 1
                continue

            audio_array = audio_array[:20 * TARGET_SR]

            try:
                instruction = (
                    f"Transcribe the following {lang} audio accurately. "
                    "Output only the transcription text, nothing else."
                )
                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an assistant that transcribes speech accurately."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": (audio_array, TARGET_SR)},
                            {"type": "text",  "text":  instruction}
                        ]
                    }
                ]

                prompt = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True
                )
                inputs = self.processor(
                    audio=[audio_array],
                    text=[prompt],
                    sampling_rate=TARGET_SR,
                    return_tensors="pt",
                    padding=True
                ).to("cuda")

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        use_cache=True,
                        do_sample=False,
                        pad_token_id=self.processor.pad_token_id,
                        eos_token_id=self.processor.eos_token_id,
                    )
                    input_len = inputs["input_ids"].shape[1]
                    pred = self.processor.decode(
                        output[0][input_len:], skip_special_tokens=True
                    ).strip()

                preds.setdefault(lang, []).append(pred)
                refs.setdefault(lang,  []).append(entry["text"])

            except Exception as e:
                print(f"Inference error: {e}")
                skipped += 1
                continue

        model.train()

        if not preds:
            print(f"⚠️  WER: all {skipped} samples skipped.")
            return None

        return preds, refs

    def on_evaluate(self, args, state, control, model, **kwargs):
        result = self.compute_wer(model)
        if result is None:
            return

        preds, refs = result
        log_dict = {}

        for lang in preds:
            wer = jiwer.wer(refs[lang], preds[lang])
            log_dict[f"eval/wer_{lang.lower()}"] = wer
            print(f"\n[WandB] Step {state.global_step} | WER {lang}: {wer:.4f} ({wer*100:.1f}%)")

        wandb.log(log_dict, step=state.global_step)


# ── Data helpers ──────────────────────────────────────────────────────────────
def process_stream_to_disk(dataset, lang, n_samples, audio_dir, tag, prefix, audio_col="audio"):
    entries = []
    skipped = 0

    try:
        dataset = dataset.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))
    except Exception as e:
        print(f"⚠️  Could not cast column '{audio_col}': {e}")
        return entries

    stream = iter(dataset)
    pbar   = tqdm(stream, desc=f"Processing {lang} {tag}", total=n_samples)

    for sample in pbar:
        if len(entries) >= n_samples:
            break

        text = (
            sample.get("normalized") or
            sample.get("verbatim") or
            sample.get("transcript") or
            sample.get("transcription") or
            sample.get("text") or ""
        ).strip()
        if not text:
            skipped += 1
            continue

        try:
            audio_data = sample.get(audio_col)
            if audio_data is None:
                skipped += 1
                continue

            array = audio_data["array"]
            if array is None:
                skipped += 1
                continue
            array = np.array(array, dtype=np.float32)

            if array.ndim > 1:
                array = array.mean(axis=0)

        except Exception as e:
            print(f"Audio extract error: {e}")
            skipped += 1
            continue

        duration = len(array) / TARGET_SR
        if duration > 30.0 or duration < 0.5:
            skipped += 1
            pbar.set_postfix(saved=len(entries), skipped=skipped)
            continue

        wav_path = os.path.join(audio_dir, f"{prefix}_{tag}_{len(entries):06d}.wav")
        try:
            sf.write(wav_path, array, TARGET_SR)
        except Exception as e:
            print(f"Write error: {e}")
            skipped += 1
            continue

        entries.append({"audio": wav_path, "text": text, "lang": lang.capitalize()})
        pbar.set_postfix(saved=len(entries), skipped=skipped)

    print(f"  ✅ {lang} {tag}: {len(entries)} saved, {skipped} skipped")
    return entries


# ── Dataset ───────────────────────────────────────────────────────────────────
class ASRDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        sample      = self.entries[idx]
        audio_array, _ = sf.read(sample["audio"], dtype="float32")
        lang        = sample.get("lang", "Tamil")
        instruction = (
            f"Transcribe the following {lang} audio accurately. "
            "Output only the transcription text, nothing else."
        )
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an assistant that transcribes speech accurately."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": (audio_array, TARGET_SR)},
                        {"type": "text",  "text":  instruction}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["text"]}]
                }
            ],
            "length": int(len(audio_array) / TARGET_SR * 50) + len(sample["text"]),
        }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # CUDA optimisations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    datasets.config.AUDIO_DECODE_BACKEND = "soundfile"

    # Auth
    from huggingface_hub import login
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("⚠️  HF_TOKEN not set — private models will fail.")
    if WB_TOKEN:
        wandb.login(key=WB_TOKEN)
    else:
        print("⚠️  WB_API_KEY not set — WandB logging will be skipped.")

    # Directories
    os.makedirs(TRAIN_AUDIO_DIR, exist_ok=True)
    os.makedirs(EVAL_AUDIO_DIR,  exist_ok=True)

    train_entries = []
    eval_entries  = []

    # ── IndicVoices (Tamil) ───────────────────────────────────────────────────
    for lang_name, fleurs_code, prefix in FLEURS_CONFIGS:
        print(f"\n--- Loading IndicVoices: {lang_name} ---")

        train_ds = load_dataset("ai4bharat/IndicVoices", fleurs_code, split="train", streaming=True)
        train_entries.extend(process_stream_to_disk(
            train_ds, lang_name, TRAIN_SAMPLES[lang_name],
            TRAIN_AUDIO_DIR, "train", prefix, audio_col="audio_filepath"
        ))
        gc.collect(); torch.cuda.empty_cache()

        eval_ds = load_dataset("ai4bharat/IndicVoices", fleurs_code, split="valid", streaming=True)
        eval_entries.extend(process_stream_to_disk(
            eval_ds, lang_name, EVAL_SAMPLES[lang_name],
            EVAL_AUDIO_DIR, "eval", prefix, audio_col="audio_filepath"
        ))
        gc.collect(); torch.cuda.empty_cache()

    # ── Svarah (English) ──────────────────────────────────────────────────────
    for lang_name, config_code, prefix in SVARAH_CONFIGS:
        try:
            print(f"\n--- Loading Svarah: {lang_name} ---")

            train_ds = load_dataset("ai4bharat/Svarah", split="test", streaming=True)
            train_entries.extend(process_stream_to_disk(
                train_ds, lang_name, TRAIN_SAMPLES.get(lang_name, 50),
                TRAIN_AUDIO_DIR, "train", prefix, audio_col="audio_filepath"
            ))

            eval_ds = load_dataset("ai4bharat/Svarah", split="test", streaming=True)
            eval_entries.extend(process_stream_to_disk(
                eval_ds, lang_name, EVAL_SAMPLES.get(lang_name, 10),
                EVAL_AUDIO_DIR, "eval", prefix, audio_col="audio_filepath"
            ))

        except Exception as e:
            print(f"❌ Svarah load error: {e}")

    random.shuffle(train_entries)
    print(f"\nTrain: {len(train_entries)}, Eval: {len(eval_entries)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_PATH,
        use_gradient_checkpointing="unsloth",
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, "gemma-4")

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "post", "linear_start", "linear_end",
        "embedding_projection",
        "ffw_layer_1", "ffw_layer_2",
        "output_proj",
    ]
    )

    # ── Training ──────────────────────────────────────────────────────────────
    wandb.init(project="gemma4-asr-ft", name=RUN_NAME)

    metrics_callback = WandbMetricsCallback(
        processor=tokenizer,
        eval_entries=eval_entries,
        sample_size=len(eval_entries)
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ASRDataset(train_entries),
        eval_dataset=ASRDataset(eval_entries),
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        callbacks=[PrintLossCallback(), metrics_callback],
        args=SFTConfig(
            output_dir=f"./{RUN_NAME}",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            optim="adamw_8bit",
            # optim="adamw_torch_fused",
            # fp16=True,
            bf16=True,
            tf32=True,  
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            logging_steps=50,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            # load_best_model_at_end=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_length=MAX_SEQ_LENGTH,
            length_column_name="length",
            report_to="all",
            push_to_hub=True,
            hub_strategy="checkpoint",
            hub_model_id=HUB_MODEL_ID,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )

    trainer.train()
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()
    print("✅ Training complete!")

    # ── Verify upload ─────────────────────────────────────────────────────────
    from huggingface_hub import list_repo_files
    files = list(list_repo_files(HUB_MODEL_ID, token=HF_TOKEN))
    print(files)


if __name__ == "__main__":
    main()