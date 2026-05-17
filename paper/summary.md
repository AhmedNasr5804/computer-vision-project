# Project numerical summary

## Eye pipeline (Samsung S24 Ultra target)

- classical HOG+SVM     acc 0.8736  f1 0.8763
- custom CNN (scratch)  acc 0.8956  f1 0.8984  (25,778 params)
- mobilenetv3_small_frozen        acc 0.8516  f1 0.8586  (940,274 params)
- mobilenetv3_small_finetune      acc 0.8434  f1 0.8472  (940,274 params)
- mobilenetv2_frozen              acc 0.8736  f1 0.8715  (2,260,546 params)
- mobilenetv2_finetune            acc 0.8846  f1 0.8877  (2,260,546 params)
- WINNER: **lw_wide**

## Lane pipeline (Raspberry Pi 4 target)

- classical Hough+SVM   acc 0.5050  f1 0.5750
- custom CNN (scratch)  acc 0.7023  f1 0.6592  (86,276 params)
- mobilenetv2_frozen              acc 0.7241  f1 0.7379  (711,348 params)
- mobilenetv2_finetune            acc 0.7759  f1 0.7785  (711,348 params)
- mobilenetv3_small_frozen        acc 0.6070  f1 0.6428  (584,892 params)
- mobilenetv3_small_finetune      acc 0.7107  f1 0.6819  (584,892 params)
- WINNER: **lw_baseline**

- Pi fine-tune: acc 0.3939 -> 1.0000

## Deployment
- Eye PTQ-int8: 106 KB, 0.43 ms local CPU
- Lane PTQ-int8: 98 KB, 0.55 ms local CPU 4thr