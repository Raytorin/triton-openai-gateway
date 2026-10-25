# Pinned upstream Qwen chat templates

These UTF-8 templates come from public `Qwen/*` Hugging Face repositories at
immutable revisions. Each folder's `metadata.json` records the exact revision,
source URL and SHA256, extracted template SHA256 and Apache-2.0 declaration URL.
For `tokenizer_config.json`, only its `chat_template` string was extracted using
`json.loads(file_bytes)["chat_template"].encode("utf-8")`, with no extra newline.
For `chat_template.jinja`, the file is retained byte for byte.

The README frontmatter at the same pinned revision declares `license: apache-2.0`.
`LICENSE` is the standard Apache License 2.0 text (the repositories do not all
contain a standalone license file); its SHA256 is recorded separately. Retained
upstream template text is unmodified. SHA256 of the source JSON and README are
recorded for independent retrieval verification.

To refresh deliberately, download just the indicated file and README with
`huggingface_hub.hf_hub_download(repo_id, filename, revision=exact_commit)` and
recompute the metadata. Do not download model weights or full tokenizer files.
No network access occurs during pytest. Fixtures are neither a vocabulary nor
weights and do not prove inference quality, runtime support, or token budgets.

`test_qwen_templates.py` renders these files in Jinja2's immutable sandbox with
the loopcontrols extension and JSON/exception globals used by Hugging Face.
Transformers is absent from the isolated test environment, so this minimal
facade verifies actual Jinja behavior, not the entire HF tokenizer integration.
The real LiteLLM HTTP checks remain in `tests/client_compatibility.py`.
