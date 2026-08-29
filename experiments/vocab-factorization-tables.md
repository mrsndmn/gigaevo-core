# Vocabulary factorization — measured numbers

Generated snapshot of every metric in the study. The underlying JSON lives under `results/`, which is gitignored; this file is the tracked record. Interpretation and conclusions live in `experiments/vocab-factorization.md` — this file is data only.

- Generated: 2026-08-30 01:43
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
| orig-frozen | native | 0.7796 | 0.7796 | 0.1562 | 0.6109 | 0.0446 | 0.2906 | 0.1089 | 1.0355 | 151,936 | 311,164,928 | 1,720,574,976 | 0 | 0.0 |
| orig-reinit | random | 4.8962 | 1.4281 | 0.1583 | 0.6792 | 0.1621 | 3.3129 | 0.5793 | 1.8156 | 151,936 | 311,164,928 | 1,720,574,976 | 807 | 69.4 |
| bpe8k-t-random | random | 4.7938 | 1.4098 | 0.1387 | 0.6014 | 0.1124 | 1.5489 | 0.4985 | 1.6426 | 8,000 | 16,384,000 | 1,425,794,048 | 853 | 51.0 |
| bpe8k-t | mean_of_pieces | 2.7704 | 0.7704 | 0.1437 | 0.5911 | 0.0548 | 0.7052 | 0.1975 | 1.0129 | 8,000 | 16,384,000 | 1,425,794,048 | 853 | 51.0 |
| fact8k-nosuf-t | mean_of_pieces | 4.2587 | 0.8175 | 0.1544 | 0.6367 | 0.0167 | 0.4964 | 0.0751 | 2.0175 | 8,000 | 16,416,768 | 1,425,826,816 | 898 | 53.9 |
| bpe8k-t-unfrozen | mean_of_pieces | 2.7704 | 0.7869 | 0.1756 | 0.7460 | 0.1327 | 1.1742 | 0.3472 | 1.5967 | 8,000 | 16,384,000 | 1,425,794,048 | 853 | 74.8 |
| shrink0-mask | shrink | 0.7796 | 0.7796 | 0.1562 | 0.6109 | 0.0447 | 0.2906 | 0.1089 | 1.0355 | 74,957 | 153,511,936 | 1,562,921,984 | 0 | 0.0 |
| shrink1-case | shrink | 1.5353 | 0.7757 | 0.1574 | 0.6374 | 0.0330 | 1.1448 | 0.1344 | 1.2058 | 48,703 | 99,768,320 | 1,509,178,368 | 807 | 62.9 |
| shrink2-tail100 | shrink | 1.9321 | 0.7775 | 0.1537 | 0.6292 | 0.0280 | 1.1461 | 0.1393 | 1.2102 | 17,385 | 35,629,056 | 1,445,039,104 | 821 | 54.1 |
| shrink3-tail1k | shrink | 5.4405 | 0.8144 | 0.1546 | 0.6290 | 0.0265 | 1.0951 | 0.1409 | 1.1796 | 4,783 | 9,820,160 | 1,419,230,208 | 920 | 57.8 |
| tree8k-x2 | mean_of_pieces | 3.0625 | 1.2281 | 0.1551 | 0.6576 | 0.1197 | 1.6521 | 0.5333 | 1.7352 | 8,000 | 8,024,064 | 1,417,434,112 | 853 | 80.3 |
| tree8k-x2-fgb | mean_of_pieces | 8.1113 | 1.4913 | 0.1368 | 0.5926 | 0.1134 | 1.6013 | 0.5337 | 1.7351 | 8,000 | 7,929,856 | 1,417,339,904 | 853 | 73.4 |
| tree8k-x2-film | mean_of_pieces | 3.2231 | 1.7287 | 0.1112 | 0.4979 | 0.0820 | 1.9486 | 0.5634 | 1.5531 | 8,000 | 8,060,928 | 1,417,470,976 | 853 | 113.9 |
| tree8k-x4 | mean_of_pieces | 3.1167 | 1.7270 | 0.1139 | 0.5110 | 0.0896 | 1.9313 | 0.5674 | 1.5434 | 8,000 | 4,063,232 | 1,413,473,280 | 853 | 82.0 |

## 6. Under the active-vocabulary mask (W0 / P0)

Only ids the corpus exercises stay in the stem softmax — in training and in evaluation, identically for every arm — so a per-arm BPB delta is about the arm's embedding rather than about how its rows happen to sit near the half of the host table this ru+en corpus never touches. `d ... vs unmasked` compares against the same harness with the mask lifted; masking removes terms from the normalizer, so it can only lower BPB, and the question is whether it does so by more than the 0.0016 noise floor. Compression ratios downstream are quoted against `flat floats`.

| experiment | active / table | flat floats | clean masked | typo1 masked | typo5 masked | lower masked | upper masked | title masked | randcase masked | d clean vs unmasked | d typo1 vs unmasked | d typo5 vs unmasked | d lower vs unmasked | d upper vs unmasked | d title vs unmasked | d randcase vs unmasked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| orig-frozen | 74957 / 151936 | 153.5M | 0.7796 | 0.9358 | 1.3905 | 0.8243 | 1.0702 | 0.8885 | 1.8151 | - | - | - | - | - | - | - |
| orig-reinit | 74957 / 151936 | 153.5M | 1.4281 | 1.5864 | 2.1073 | 1.5902 | 4.7410 | 2.0075 | 3.2438 | - | - | - | - | - | - | - |
| bpe8k-t-random | 7949 / 8000 | 16.3M | 1.4098 | 1.5485 | 2.0112 | 1.5222 | 2.9587 | 1.9083 | 3.0524 | - | - | - | - | - | - | - |
| bpe8k-t | 7949 / 8000 | 16.3M | 0.7704 | 0.9141 | 1.3615 | 0.8252 | 1.4756 | 0.9679 | 1.7833 | - | - | - | - | - | - | - |
| fact8k-nosuf-t | 7951 / 8000 | 16.3M | 0.8175 | 0.9719 | 1.4542 | 0.8342 | 1.3139 | 0.8926 | 2.8350 | - | - | - | - | - | - | - |
| bpe8k-t-unfrozen | 7949 / 8000 | 16.3M | 0.7869 | 0.9626 | 1.5329 | 0.9197 | 1.9611 | 1.1342 | 2.3837 | - | - | - | - | - | - | - |
| shrink0-mask | ? / ? | 0.0M | 0.7796 | 0.9359 | 1.3905 | 0.8243 | 1.0702 | 0.8886 | 1.8152 | - | - | - | - | - | - | - |
| shrink1-case | ? / ? | 0.0M | 0.7757 | 0.9331 | 1.4131 | 0.8087 | 1.9205 | 0.9101 | 1.9814 | - | - | - | - | - | - | - |
| shrink2-tail100 | ? / ? | 0.0M | 0.7775 | 0.9312 | 1.4067 | 0.8055 | 1.9236 | 0.9168 | 1.9877 | - | - | - | - | - | - | - |
| shrink3-tail1k | ? / ? | 0.0M | 0.8144 | 0.9690 | 1.4434 | 0.8408 | 1.9094 | 0.9553 | 1.9939 | - | - | - | - | - | - | - |
| tree8k-x2 | 7949 / 8000 | 16.3M | 1.2281 | 1.3832 | 1.8857 | 1.3478 | 2.8801 | 1.7614 | 2.9632 | - | - | - | - | - | - | - |
| tree8k-x2-fgb | 7949 / 8000 | 16.3M | 1.4913 | 1.6280 | 2.0838 | 1.6047 | 3.0926 | 2.0249 | 3.2263 | - | - | - | - | - | - | - |
| tree8k-x2-film | 7949 / 8000 | 16.3M | 1.7287 | 1.8399 | 2.2266 | 1.8107 | 3.6774 | 2.2921 | 3.2818 | - | - | - | - | - | - | - |
| tree8k-x4 | 7949 / 8000 | 16.3M | 1.7270 | 1.8409 | 2.2379 | 1.8166 | 3.6582 | 2.2943 | 3.2704 | - | - | - | - | - | - | - |
