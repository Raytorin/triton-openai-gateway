#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare local DeepSeek prompt token IDs with an independently pinned reference."""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'gateway').is_dir():
    ROOT = Path('/opt/tritonserver')
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_config', type=Path, help='local model.json with absolute checkpoint/tokenizer paths and revision')
    parser.add_argument('--reference', type=Path, required=True,
                        help='reviewed JSON reference with artifact hashes, independent prompts and literal token IDs')
    parser.add_argument('--json', action='store_true', help='print mismatch locations/counts without conversation text')
    parser.add_argument('--model-config-dir', type=Path,
                        help='explicit absolute local directory containing gateway.json model-local reasoning controls')
    args = parser.parse_args()
    try:
        from gateway.deepseek_tokenizer_validation import read_reference, validate_tokenizer
        config = read_reference(args.model_config)
        reference = read_reference(args.reference)
        report = validate_tokenizer(config, reference, model_config_dir=args.model_config_dir)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f'token qualification failed: {error}', file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"prompt_token_id_parity_only: {report['case_count']} cases; matched={report['matched']}; "
              f"reference_kind={report['reference_kind']}; not engine or GPU qualification")
        for case in report['cases']:
            if not case['matched']:
                print(f"{case['name']}: prompt={case['prompt_match']} token_ids={case['token_ids_match']} "
                      f"first_difference={case['first_token_difference']}")
        for field, mismatch in report['special_token_mismatches'].items():
            print(f'{field}: {mismatch}')
    return 0 if report['matched'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
