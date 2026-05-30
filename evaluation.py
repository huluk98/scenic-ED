import os
import json
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================
# RUN COMMANDS
# ============================================================
# Single GPU:
# python evaluation.py
#
# 8 GPUs:
# torchrun --nproc_per_node=8 evaluation.py
# ============================================================


# ============================================================
# CONFIG PATHS
# ============================================================
MODEL_PATH = "./sft"

JSON1_PATH = "/nvme1/home/luke/Encoder-Chinese-SLM/data/scenic/SCENIC_full_training_dataset.json"
JSON2_PATH = "/nvme1/home/luke/Encoder-Chinese-SLM/data/scenic/YOUR_SECOND_EVAL_FILE.json"

OUTPUT_DIR = "./eval_outputs"

BATCH_SIZE = 8
MAX_INPUT_LEN = 256
MAX_NEW_TOKENS = 128
IGNORE_SPACES = False


def setup_distributed():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        return True, rank, world_size, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def normalize_text(text):
    text = str(text).strip()
    if IGNORE_SPACES:
        text = "".join(text.split())
    return text


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of JSON objects.")

    return data


def shard_data(data, rank, world_size):
    return data[rank::world_size]


@torch.no_grad()
def evaluate_file(model, tokenizer, data, device):
    model.eval()

    total = 0
    pass1_correct = 0
    pass5_correct = 0
    outputs = []

    for start in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[start:start + BATCH_SIZE]

        prompts = [item["prompt"] for item in batch]
        targets = [item["response"] for item in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
            if k != "token_type_ids"
        }

        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=5,
            num_return_sequences=5,
            do_sample=False,
            early_stopping=True,
        )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for i, item in enumerate(batch):
            pred_5 = decoded[i * 5:(i + 1) * 5]
            pred_1 = pred_5[0]

            gold = normalize_text(targets[i])
            pred_1_norm = normalize_text(pred_1)
            pred_5_norm = [normalize_text(p) for p in pred_5]

            is_pass1 = pred_1_norm == gold
            is_pass5 = gold in pred_5_norm

            pass1_correct += int(is_pass1)
            pass5_correct += int(is_pass5)
            total += 1

            outputs.append({
                "prompt": item["prompt"],
                "target": targets[i],
                "pass1_prediction": pred_1,
                "pass5_predictions": pred_5,
                "pass1_correct": is_pass1,
                "pass5_correct": is_pass5,
            })

    return {
        "total": total,
        "pass1_correct": pass1_correct,
        "pass5_correct": pass5_correct,
        "pass1_accuracy": pass1_correct / total * 100 if total else 0.0,
        "pass5_accuracy": pass5_correct / total * 100 if total else 0.0,
        "outputs": outputs,
    }


def gather_results(local_result, distributed, rank, world_size):
    if not distributed:
        return local_result

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_result)

    if rank != 0:
        return None

    merged = {
        "total": 0,
        "pass1_correct": 0,
        "pass5_correct": 0,
        "outputs": [],
    }

    for result in gathered:
        merged["total"] += result["total"]
        merged["pass1_correct"] += result["pass1_correct"]
        merged["pass5_correct"] += result["pass5_correct"]
        merged["outputs"].extend(result["outputs"])

    merged["pass1_accuracy"] = merged["pass1_correct"] / merged["total"] * 100
    merged["pass5_accuracy"] = merged["pass5_correct"] / merged["total"] * 100

    return merged


def evaluate_dataset(name, path, model, tokenizer, rank, world_size, distributed, device):
    full_data = load_json(path)
    local_data = shard_data(full_data, rank, world_size)

    if rank == 0:
        print(f"\nEvaluating {name}")
        print(f"Path: {path}")
        print(f"Total samples: {len(full_data)}")

    local_result = evaluate_file(
        model=model,
        tokenizer=tokenizer,
        data=local_data,
        device=device,
    )

    merged_result = gather_results(
        local_result=local_result,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        output_path = os.path.join(OUTPUT_DIR, f"{name}_predictions.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_result["outputs"], f, ensure_ascii=False, indent=2)

        print(f"{name} Pass@1: {merged_result['pass1_accuracy']:.2f}%")
        print(f"{name} Pass@5: {merged_result['pass5_accuracy']:.2f}%")
        print(f"Saved predictions to: {output_path}")

        return {
            "file": path,
            "total": merged_result["total"],
            "pass1_correct": merged_result["pass1_correct"],
            "pass5_correct": merged_result["pass5_correct"],
            "pass1_accuracy": merged_result["pass1_accuracy"],
            "pass5_accuracy": merged_result["pass5_accuracy"],
        }

    return None


def main():
    distributed, rank, world_size, device = setup_distributed()

    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Using world size: {world_size}")
        print(f"Using device: {device}")
        print(f"Loading model from: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.to(device)

    summary = {}

    result1 = evaluate_dataset(
        name="eval_file_1",
        path=JSON1_PATH,
        model=model,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        device=device,
    )

    result2 = evaluate_dataset(
        name="eval_file_2",
        path=JSON2_PATH,
        model=model,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        device=device,
    )

    if rank == 0:
        summary["eval_file_1"] = result1
        summary["eval_file_2"] = result2

        summary_path = os.path.join(OUTPUT_DIR, "summary.json")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\nFinal Summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nSaved summary to: {summary_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
