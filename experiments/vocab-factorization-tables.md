# Vocabulary factorization — measured numbers

Generated snapshot of every metric in the study. The underlying JSON lives under `results/`, which is gitignored; this file is the tracked record. Interpretation and conclusions live in `experiments/vocab-factorization.md` — this file is data only.

- Generated: 2026-08-28 03:50
- Transplant host: `Qwen/Qwen3-1.7B`
- Corpus: ru+en Wikipedia, NFC, document-hash split (`data.py`)
- Regenerate: `python -m experiments.vocab_factorization.report --out experiments/vocab-factorization-tables.md`

Measured run-to-run noise (bpe8k trained twice, identical config): **0.0016 BPB**. Treat smaller differences as nothing.

## 1. Training-free screen

Absolute tokens/byte per suite, *not* the inflation ratio: inflation is relative to each arm's own clean baseline and so rewards a bad baseline (it ranked `fact8k-nosuf` best on randcase purely because its clean tokens/byte was worst).

| arm | vocab | tok/byte | pieces/word | whole-word % | tpb UPPER | tpb Title | tpb randcase | tpb typo5 | embed params (d=512) | lossless |
|---|---|---|---|---|---|---|---|---|---|---|
| bpe32k | 32,000 | 0.1751 | 1.5400 | 64.6 | 0.4849 | 0.2194 | 0.2873 | 0.2088 | 16,384,000 | yes |
| bpe16k | 16,000 | 0.1961 | 1.8020 | 52.2 | 0.5305 | 0.2439 | 0.3051 | 0.2274 | 8,192,000 | yes |
| bpe8k | 8,000 | 0.2231 | 2.1440 | 41.0 | 0.5789 | 0.2718 | 0.3272 | 0.2510 | 4,096,000 | yes |
| byte | 256 | 1.0000 | 13.5180 | 0.2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 131,072 | yes |
| fact16k | 16,000 | 0.1864 | 1.4330 | 78.9 | 0.1864 | 0.1864 | 0.2541 | 0.2190 | 8,327,680 | yes |
| fact8k | 8,000 | 0.2158 | 1.7670 | 65.4 | 0.2158 | 0.2158 | 0.2737 | 0.2443 | 4,231,680 | yes |
| fact16k-nosuf | 16,000 | 0.2135 | 1.8160 | 29.6 | 0.2135 | 0.2135 | 0.2725 | 0.2371 | 8,196,096 | yes |
| fact8k-nosuf | 8,000 | 0.2355 | 2.0160 | 24.4 | 0.2354 | 0.2354 | 0.2887 | 0.2571 | 4,100,096 | yes |
| fact8k-freqfix | 8,000 | 0.2096 | 1.8750 | 47.7 | 0.2093 | 0.2093 | 0.4001 | 0.2377 | 4,100,096 | yes |
| fact8k-v2 | 8,000 | 0.2093 | 1.8750 | 47.7 | 0.2093 | 0.2093 | 0.2760 | 0.2367 | 4,100,096 | yes |

## 2. From-scratch 33M grid

### Absolute bits per byte

| experiment | BPB clean | BPB typo1 | BPB typo5 | BPB lower | BPB upper | BPB title | BPB randcase |
|---|---|---|---|---|---|---|---|
| bpe16k | 1.1919 | 1.3508 | 1.9057 | 1.4157 | 3.1684 | 1.8533 | 3.2126 |
| fact16k | 1.4610 | 1.6131 | 2.1368 | 1.5570 | 2.1716 | 1.7641 | 3.4358 |
| bpe32k | 1.2039 | 1.3625 | 1.9103 | 1.4254 | 3.2803 | 1.8621 | 3.1680 |
| fact16k-nosuf | 1.2137 | 1.3823 | 1.9627 | 1.3201 | 1.9365 | 1.5248 | 3.2656 |
| fact8k | 1.3911 | 1.5478 | 2.0892 | 1.5093 | 2.2108 | 1.7200 | 3.4040 |
| bpe8k | 1.1881 | 1.3449 | 1.9044 | 1.4182 | 3.1254 | 1.8584 | 3.2392 |
| fact16k-ngram | 1.4565 | 1.6097 | 2.1352 | 1.5504 | 2.1618 | 1.7611 | 3.4295 |
| bpe16k-ngram | 1.2143 | 1.3704 | 1.9204 | 1.4572 | 3.3089 | 1.9300 | 3.2478 |
| byte | 1.1110 | 1.2787 | 1.8822 | 1.3259 | 3.0107 | 1.6640 | 2.9458 |
| fact8k-nosuf | 1.2074 | 1.3770 | 1.9666 | 1.3199 | 2.0463 | 1.5371 | 3.7161 |
| fact8k-freqfix | 1.2060 | 1.3747 | 1.9555 | 1.2989 | 1.9388 | 1.4977 | 3.2338 |
| fact8k-v2 | 1.2056 | 1.3749 | 1.9549 | 1.3004 | 1.8969 | 1.5088 | 3.2984 |

### Delta vs clean

| experiment | d typo1 | d typo5 | d lower | d upper | d title | d randcase |
|---|---|---|---|---|---|---|
| bpe16k | 0.1589 | 0.7138 | 0.2238 | 1.9765 | 0.6615 | 2.0207 |
| fact16k | 0.1521 | 0.6758 | 0.0959 | 0.7106 | 0.3030 | 1.9747 |
| bpe32k | 0.1586 | 0.7065 | 0.2216 | 2.0764 | 0.6583 | 1.9641 |
| fact16k-nosuf | 0.1686 | 0.7490 | 0.1064 | 0.7228 | 0.3112 | 2.0519 |
| fact8k | 0.1567 | 0.6981 | 0.1181 | 0.8196 | 0.3289 | 2.0129 |
| bpe8k | 0.1568 | 0.7163 | 0.2301 | 1.9373 | 0.6703 | 2.0511 |
| fact16k-ngram | 0.1532 | 0.6787 | 0.0939 | 0.7053 | 0.3046 | 1.9730 |
| bpe16k-ngram | 0.1560 | 0.7061 | 0.2429 | 2.0946 | 0.7157 | 2.0335 |
| byte | 0.1678 | 0.7712 | 0.2150 | 1.8997 | 0.5530 | 1.8348 |
| fact8k-nosuf | 0.1696 | 0.7592 | 0.1125 | 0.8389 | 0.3297 | 2.5087 |
| fact8k-freqfix | 0.1686 | 0.7495 | 0.0928 | 0.7327 | 0.2917 | 2.0278 |
| fact8k-v2 | 0.1694 | 0.7494 | 0.0948 | 0.6913 | 0.3032 | 2.0928 |

### Clean BPB split by prediction head

| experiment | BPB total | stem | role | suffix |
|---|---|---|---|---|
| bpe16k | 1.1919 | 1.1919 | - | - |
| fact16k | 1.4610 | 1.1181 | 0.0576 | 0.2854 |
| bpe32k | 1.2039 | 1.2039 | - | - |
| fact16k-nosuf | 1.2137 | 1.1519 | 0.0617 | - |
| fact8k | 1.3911 | 1.1186 | 0.0598 | 0.2127 |
| bpe8k | 1.1881 | 1.1881 | - | - |
| fact16k-ngram | 1.4565 | 1.1146 | 0.0573 | 0.2845 |
| bpe16k-ngram | 1.2143 | 1.2143 | - | - |
| byte | 1.1110 | 1.1110 | - | - |
| fact8k-nosuf | 1.2074 | 1.1426 | 0.0648 | - |
| fact8k-freqfix | 1.2060 | 1.1423 | 0.0637 | - |
| fact8k-v2 | 1.2056 | 1.1415 | 0.0641 | - |

### Cost and capacity

| experiment | BPB clean | total params | embed params | tok/byte | context bytes | steps | train tokens | minutes |
|---|---|---|---|---|---|---|---|---|
| bpe16k | 1.1919 | 33,375,232 | 8,192,000 | 0.2096 | 4,884 | 2,013 | 87,986,187 | 13.5 |
| fact16k | 1.4610 | 33,646,592 | 8,463,360 | 0.1991 | 5,144 | 1,912 | 83,545,403 | 13.2 |
| bpe32k | 1.2039 | 41,567,232 | 16,384,000 | 0.1876 | 5,459 | 1,801 | 78,725,113 | 15.0 |
| fact16k-nosuf | 1.2137 | 33,383,424 | 8,200,192 | 0.2249 | 4,554 | 2,159 | 94,369,720 | 14.7 |
| fact8k | 1.3911 | 29,550,592 | 4,367,360 | 0.2291 | 4,470 | 2,200 | 96,152,841 | 13.7 |
| bpe8k | 1.1881 | 29,279,232 | 4,096,000 | 0.2374 | 4,312 | 2,280 | 99,653,699 | 13.8 |
| fact16k-ngram | 1.4565 | 34,203,648 | 9,020,416 | 0.1991 | 5,144 | 1,912 | 83,545,403 | 13.7 |
| bpe16k-ngram | 1.2143 | 33,932,288 | 8,749,056 | 0.2096 | 4,884 | 2,013 | 87,986,187 | 14.0 |
| byte | 1.1110 | 25,314,304 | 131,072 | 1.0000 | 1,024 | 9,605 | 419,683,874 | 51.9 |
| fact8k-nosuf | 1.2074 | 29,287,424 | 4,104,192 | 0.2488 | 4,116 | 2,389 | 104,413,105 | 14.6 |
| fact8k-freqfix | 1.2060 | 29,287,424 | 4,104,192 | 0.2213 | 4,628 | 2,125 | 92,856,668 | 13.1 |
| fact8k-v2 | 1.2056 | 29,287,424 | 4,104,192 | 0.2209 | 4,636 | 2,121 | 92,698,932 | 13.0 |

## 3. Corrupted-context BPB

Corruption confined to unscored context; the denominator is clean-suffix bytes only, so nothing is charged for emitting noise. `mismatch` replaces the context with an unrelated document and calibrates how much any context damage can cost.

| experiment | ctx BPB clean | d typo1 | d typo5 | d typo10 | d randcase | d mismatch | typo10 / mismatch |
|---|---|---|---|---|---|---|---|
| bpe8k | 1.1754 | 0.0005 | 0.0024 | 0.0045 | 0.0081 | 0.0262 | 17.1% |
| fact8k-nosuf | 1.1932 | 0.0005 | 0.0019 | 0.0039 | 0.0074 | - | - |
| fact8k-freqfix | 1.1939 | 0.0008 | 0.0038 | 0.0068 | 0.0194 | - | - |
| fact8k-v2 | 1.1939 | 0.0008 | 0.0035 | 0.0066 | 0.0103 | 0.0269 | 24.6% |

## 4. Transplant onto a frozen Qwen3-1.7B

Body frozen; only the embedding table and heads train.

| experiment | init | BPB step0 | BPB final | d typo1 | d typo5 | d lower | d upper | d title | d randcase | vocab | embed params | total params | steps | minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| orig-frozen | native | 0.7125 | 0.7125 | 0.1463 | 0.5781 | 0.0392 | 0.3053 | 0.1184 | 1.0014 | 151,936 | 311,164,928 | 1,720,574,976 | 0 | 0.0 |
| orig-reinit | random | 4.9866 | 1.3318 | 0.1385 | 0.6093 | 0.1181 | 2.7703 | 0.5339 | 1.6804 | 151,936 | 311,164,928 | 1,720,574,976 | 807 | 65.5 |
| bpe8k-t-random | random | 4.3854 | 1.3130 | 0.1266 | 0.5511 | 0.0887 | 1.5170 | 0.4860 | 1.5324 | 8,000 | 16,384,000 | 1,425,794,048 | 853 | 50.9 |
| bpe8k-t | mean_of_pieces | 2.7078 | 0.7031 | 0.1359 | 0.5584 | 0.0469 | 0.6979 | 0.1965 | 0.9744 | 8,000 | 16,384,000 | 1,425,794,048 | 853 | 50.9 |
| fact8k-nosuf-t | mean_of_pieces | 4.1309 | 0.7390 | 0.1475 | 0.6011 | 0.0149 | 0.4599 | 0.0725 | 1.7958 | 8,000 | 16,416,768 | 1,425,826,816 | 898 | 53.7 |
