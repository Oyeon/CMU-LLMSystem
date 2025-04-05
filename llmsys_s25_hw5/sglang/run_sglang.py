# run_sglang.py

import argparse
import json
from tqdm import tqdm
from datasets import load_dataset

import sglang as sgl  # Will load the package from myproj/sglang/__init__.py

def main():
    parser = argparse.ArgumentParser(description="Run inference with a specific model path.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
        help="A dummy default model path."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs.jsonl",
        help="Where to write JSON lines of results."
    )
    args = parser.parse_args()

    # Example: load a small dataset. Adjust to your real dataset as needed.
    dataset = load_dataset(
        "tatsu-lab/alpaca_eval",
        data_files="alpaca_eval.json",
        split="eval",
        trust_remote_code=True
    )

    # Create an instance of our SGLangEngine
    llm = sgl.SGLangEngine(
        model_path=args.model_path,
        dp_size=1,
        mem_fraction_static=0.3,
        use_radix_cache=True,
        use_compressed_fsm=True,
    )

    # Collect the instructions
    prompts = [item["instruction"] for item in dataset]

    # Example generation params
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 8192
    }

    outputs = []

    # Simple batching
    batch_size = 4
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
        results = llm.generate(batch_prompts, **sampling_params)
        outputs.extend(results)

    # Write to output file
    with open(args.output_file, "w") as f:
        for idx, out_text in enumerate(outputs):
            f.write(json.dumps({
                "instruction": prompts[idx],
                "output": out_text
            }) + "\n")

if __name__ == "__main__":
    main()
