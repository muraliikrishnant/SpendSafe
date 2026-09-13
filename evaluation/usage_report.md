# Token Usage and Cost Report

Generated: 2026-09-13T02:10:12.238561Z

This report summarizes model calls made by `code/main.py` for the run
that produced the submitted `output.csv`.

| Provider | Model | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Est. Cost (USD) |
|---|---|---|---|---|---|---|
| nvidia | openai/gpt-oss-20b | 200 | 79347 | 70924 | 150271 | $0.0584 |
| nvidia | meta/llama-3.2-11b-vision-instruct | 12 | 69205 | 1347 | 70552 | $0.0146 |
| ollama | qwen3-vl:4b | 1 | 1506 | 600 | 2106 | $0.0000 |
| ollama | qwen3:14b | 2 | 688 | 728 | 1416 | $0.0000 |

## Overall

- Total model calls: 215
- Total prompt tokens: 150746
- Total completion tokens: 73599
- Total tokens: 224345
- Average tokens per request: 897.4
- Estimated total cost: $0.0731
- Estimated cost per request: $0.000292

Note: Ollama calls are local and free (cost $0); NVIDIA NIM pricing above
is indicative and should be replaced with actual invoiced rates if available.
