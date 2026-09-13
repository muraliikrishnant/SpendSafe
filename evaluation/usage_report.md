# Token Usage and Cost Report

Generated: 2026-09-13T01:47:17.622439Z

This report summarizes model calls made by `code/main.py` for the run
that produced the submitted `output.csv`.

| Provider | Model | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Est. Cost (USD) |
|---|---|---|---|---|---|---|
| nvidia | openai/gpt-oss-20b | 200 | 79190 | 70750 | 149940 | $0.0583 |
| nvidia | meta/llama-3.2-11b-vision-instruct | 12 | 69203 | 1340 | 70543 | $0.0146 |
| ollama | qwen3-vl:4b | 1 | 1506 | 600 | 2106 | $0.0000 |
| ollama | qwen3:14b | 2 | 688 | 728 | 1416 | $0.0000 |

## Overall

- Total model calls: 215
- Total prompt tokens: 150587
- Total completion tokens: 73418
- Total tokens: 224005
- Average tokens per request: 1041.9
- Estimated total cost: $0.0729
- Estimated cost per request: $0.000339

Note: Ollama calls are local and free (cost $0); NVIDIA NIM pricing above
is indicative and should be replaced with actual invoiced rates if available.
