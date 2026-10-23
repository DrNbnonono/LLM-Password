import argparse
import os
import time
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm

from src.clients.openrouter_client import OpenRouterClient
from src.clients.siliconflow_client import SiliconFlowClient
from src.prompts.builders import build_benign_prompt, build_few_shot_prompt, build_jailbreak_prompt
from src.utils.metrics import estimate_entropy, character_diversity, zxcvbn_summary
from src.utils.dataset import ensure_dataset_path, append_rows, make_row, DATASET_FIELDS


def _build_prompt(prompt_type: str, args: argparse.Namespace) -> str:
    if prompt_type == "benign":
        return build_benign_prompt(
            length=args.length,
            require_upper=not args.no_upper,
            require_lower=not args.no_lower,
            require_digits=not args.no_digits,
            require_symbols=not args.no_symbols,
            lang=args.lang,
        )
    if prompt_type == "few-shot":
        examples: List[str] = args.examples.split(";;") if args.examples else []
        return build_few_shot_prompt(examples=examples, lang=args.lang)
    if prompt_type == "jailbreak":
        return build_jailbreak_prompt(length=args.length, style_hint=args.style, lang=args.lang)
    raise ValueError(f"Unknown prompt_type: {prompt_type}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate LLM passwords and save to CSV dataset")
    parser.add_argument("--provider", choices=["openrouter", "siliconflow"], required=True)
    parser.add_argument("--model", required=True, help="Model ID for the selected provider")
    parser.add_argument("--prompt-type", choices=["benign", "few-shot", "jailbreak"], required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", default=os.path.join("data", "llm_passwords.csv"))
    parser.add_argument("--lang", choices=["zh", "en", "both"], default="both", help="Prompt language")
    parser.add_argument("--sleep-ms", type=int, default=250, help="Sleep between requests in milliseconds")

    # Benign flags
    parser.add_argument("--no-upper", action="store_true", help="Do not require uppercase letters")
    parser.add_argument("--no-lower", action="store_true", help="Do not require lowercase letters")
    parser.add_argument("--no-digits", action="store_true", help="Do not require digits")
    parser.add_argument("--no-symbols", action="store_true", help="Do not require symbols")

    # Few-shot
    parser.add_argument("--examples", default="", help="Few-shot examples separated by ';;'")

    # Jailbreak
    parser.add_argument("--style", default="", help="Optional style hint for jailbreak prompt")

    args = parser.parse_args()

    prompt = _build_prompt(args["prompt_type"] if isinstance(args, dict) else args.prompt_type, args)

    if args.provider == "openrouter":
        client = OpenRouterClient()
        generator = lambda: client.generate_passwords(
            model=args.model,
            prompt=prompt,
            count=1,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            lang=args.lang,
        )
    else:
        client = SiliconFlowClient()
        generator = lambda: client.generate_passwords(
            model=args.model,
            prompt=prompt,
            count=1,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            lang=args.lang,
        )

    csv_path = ensure_dataset_path(args.output, DATASET_FIELDS)

    success = 0
    for _ in tqdm(range(args.count), desc="Generating"):
        try:
            pwds = generator()
            if not pwds:
                time.sleep(max(args.sleep_ms, 0) / 1000.0)
                continue
            password = pwds[0]
            ent = estimate_entropy(password)
            div = character_diversity(password)
            zsum = zxcvbn_summary(password)
            row = make_row(
                password=password,
                provider=args.provider,
                model_id=args.model,
                prompt_type=args.prompt_type,
                prompt_text=prompt,
                lang=args.lang,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                length=len(password),
                char_diversity=div,
                entropy=ent,
                pattern_score=0.0,
                crack_time=zsum["crack_time"],
            )
            append_rows(csv_path, [row], fieldnames=DATASET_FIELDS)
            success += 1
        except Exception as e:  # noqa: BLE001
            print(f"warning: generation failed, continuing: {e}")
        finally:
            time.sleep(max(args.sleep_ms, 0) / 1000.0)

    print(f"Saved {success} rows to {csv_path}")


if __name__ == "__main__":
    main()
