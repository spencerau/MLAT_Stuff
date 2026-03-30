import json
import re
import time
from pathlib import Path

import requests
import yaml
from datasets import load_dataset


def load_config(path: str = "model.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_aime25():
    ds = load_dataset("math-ai/aime25", split="test")
    return list(ds)


def query(problem: str, cfg: dict) -> tuple[str, str]:
    url = f"http://localhost:{cfg['port']}/v1/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system",    "content": cfg["system_prompt"].strip()},
            {"role": "user",      "content": problem},
        ],
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "stream": True,
    }

    thinking = ""
    answer = ""
    in_thinking = False
    in_response = False

    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = line.decode("utf-8")
            if not data.startswith("data: ") or data[6:] == "[DONE]":
                continue
            delta = json.loads(data[6:])["choices"][0]["delta"]
            r = delta.get("reasoning") or ""
            c = delta.get("content") or ""

            if r:
                if not in_thinking:
                    print("------- THINKING --------")
                    in_thinking = True
                thinking += r
                print(r, end="", flush=True)

            if c:
                if not in_response:
                    if in_thinking:
                        print("\n------- END THINKING ----\n")
                    print("------- RESPONSE --------")
                    in_response = True
                answer += c
                print(c, end="", flush=True)

    if in_response:
        print("\n------- END RESPONSE ----")
    print()
    return thinking.strip(), answer.strip()


def main():
    cfg = load_config()
    problems = load_aime25()

    log_path = Path(cfg["log_dir"])
    log_path.mkdir(parents=True, exist_ok=True)
    out_file = log_path / "results.json"

    print(f"Evaluating {len(problems)} problems -> {out_file}\n")

    results = {}

    for idx, row in enumerate(problems, start=1):
        print(f"[{idx:02d}/{len(problems)}] {row['problem']}\n")

        t0 = time.time()
        thinking, answer = query(row["problem"], cfg)
        elapsed = time.time() - t0

        print(f"\n({elapsed:.1f}s)\n")

        boxed = re.search(r"\\boxed\{(\d+)\}", answer)
        extracted_answer = boxed.group(1) if boxed else None

        results[f"question_{idx}"] = {
            "question":        row["problem"],
            "thinking":        thinking,
            "raw_answer":      answer,
            "extracted_answer": extracted_answer,
            "correct_answer":  row["answer"],
        }

        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"Done. Results saved to {out_file}")


if __name__ == "__main__":
    main()
