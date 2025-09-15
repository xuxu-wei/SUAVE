# Hard benchmark: beta/InfoVAE sweep

- Split seed generator: 202352
- Train/test split random_state: 285502
- Epochs: 20
- Training samples: 1000

| setting | y1 AUROC | y2 AUROC | y3 AUROC |
| --- | --- | --- | --- |
| beta_1.0 | 0.7189 | 0.6335 | 0.5320 |
| beta_2.0 | 0.6876 | 0.6384 | 0.5034 |
| beta_4.0 | 0.7493 | 0.6423 | 0.5325 |
| info_alpha0.3_lambda1.0 | 0.6860 | 0.6189 | 0.4879 |
| info_alpha0.7_lambda1.3 | 0.6558 | 0.6683 | 0.4799 |

## beta_1.0

- beta: 1.0
- info_config: None

### y1

- AUROC (macro): 0.7249; AUROC (micro): 0.7249
- AUPRC (macro): 0.2163; AUPRC (micro): 0.2163
- Accuracy: 0.8850
- F1 macro: 0.3030
- Losses: total=2.0843, recon=1.4259, KL=0.0071, CE=0.6535, info_penalty=0.0000, weight=0.0000

### y2

- AUROC (macro): 0.6373; AUROC (micro): 0.5403
- AUPRC (macro): 0.3606; AUPRC (micro): 0.2883
- Accuracy: 0.2433
- F1 macro: 0.1777
- Losses: total=2.8113, recon=1.4294, KL=0.0078, CE=1.3764, info_penalty=0.0000, weight=0.0000

### y3

- AUROC (macro): 0.4975; AUROC (micro): 0.4975
- AUPRC (macro): 0.5144; AUPRC (micro): 0.5144
- Accuracy: 0.5000
- F1 macro: 0.1573
- Losses: total=2.1252, recon=1.4240, KL=0.0081, CE=0.6955, info_penalty=0.0000, weight=0.0000

## beta_2.0

- beta: 2.0
- info_config: None

### y1

- AUROC (macro): 0.5705; AUROC (micro): 0.5705
- AUPRC (macro): 0.1262; AUPRC (micro): 0.1262
- Accuracy: 0.6817
- F1 macro: 0.1941
- Losses: total=2.1123, recon=1.4280, KL=0.0056, CE=0.6764, info_penalty=0.0000, weight=0.0000

### y2

- AUROC (macro): 0.6297; AUROC (micro): 0.6296
- AUPRC (macro): 0.3735; AUPRC (micro): 0.3840
- Accuracy: 0.3717
- F1 macro: 0.3143
- Losses: total=2.7972, recon=1.4241, KL=0.0065, CE=1.3640, info_penalty=0.0000, weight=0.0000

### y3

- AUROC (macro): 0.5183; AUROC (micro): 0.5183
- AUPRC (macro): 0.5219; AUPRC (micro): 0.5219
- Accuracy: 0.5133
- F1 macro: 0.3333
- Losses: total=2.1298, recon=1.4268, KL=0.0057, CE=0.6950, info_penalty=0.0000, weight=0.0000

## beta_4.0

- beta: 4.0
- info_config: None

### y1

- AUROC (macro): 0.7178; AUROC (micro): 0.7178
- AUPRC (macro): 0.3097; AUPRC (micro): 0.3097
- Accuracy: 0.7983
- F1 macro: 0.2667
- Losses: total=2.1121, recon=1.4256, KL=0.0055, CE=0.6712, info_penalty=0.0000, weight=0.0000

### y2

- AUROC (macro): 0.6422; AUROC (micro): 0.6008
- AUPRC (macro): 0.3791; AUPRC (micro): 0.2917
- Accuracy: 0.2683
- F1 macro: 0.1317
- Losses: total=2.7968, recon=1.4246, KL=0.0050, CE=1.3582, info_penalty=0.0000, weight=0.0000

### y3

- AUROC (macro): 0.4946; AUROC (micro): 0.4946
- AUPRC (macro): 0.5321; AUPRC (micro): 0.5321
- Accuracy: 0.5217
- F1 macro: 0.6504
- Losses: total=2.1326, recon=1.4238, KL=0.0051, CE=0.6946, info_penalty=0.0000, weight=0.0000

## info_alpha0.3_lambda1.0

- beta: 1.0
- info_config: {'alpha': 0.3, 'lambda_': 1.0, 'kernel': 'rbf', 'kernel_bandwidth': 1.0}

### y1

- AUROC (macro): 0.6787; AUROC (micro): 0.6787
- AUPRC (macro): 0.2040; AUPRC (micro): 0.2040
- Accuracy: 0.9000
- F1 macro: 0.1667
- Losses: total=2.0705, recon=1.4215, KL=0.0074, CE=0.6453, info_penalty=-0.0000, weight=0.3000

### y2

- AUROC (macro): 0.6406; AUROC (micro): 0.5865
- AUPRC (macro): 0.3878; AUPRC (micro): 0.2859
- Accuracy: 0.2600
- F1 macro: 0.1070
- Losses: total=2.7988, recon=1.4260, KL=0.0075, CE=1.3692, info_penalty=-0.0000, weight=0.3000

### y3

- AUROC (macro): 0.5106; AUROC (micro): 0.5106
- AUPRC (macro): 0.5151; AUPRC (micro): 0.5151
- Accuracy: 0.4983
- F1 macro: 0.0131
- Losses: total=2.1259, recon=1.4256, KL=0.0075, CE=0.6967, info_penalty=-0.0000, weight=0.3000

## info_alpha0.7_lambda1.3

- beta: 1.0
- info_config: {'alpha': 0.7, 'lambda_': 1.3, 'kernel': 'rbf', 'kernel_bandwidth': 1.0}

### y1

- AUROC (macro): 0.5918; AUROC (micro): 0.5918
- AUPRC (macro): 0.1356; AUPRC (micro): 0.1356
- Accuracy: 0.1867
- F1 macro: 0.1378
- Losses: total=2.1456, recon=1.4267, KL=0.0092, CE=0.7169, info_penalty=0.0001, weight=1.0000

### y2

- AUROC (macro): 0.6589; AUROC (micro): 0.6405
- AUPRC (macro): 0.3954; AUPRC (micro): 0.3828
- Accuracy: 0.3833
- F1 macro: 0.3501
- Losses: total=2.7870, recon=1.4241, KL=0.0101, CE=1.3603, info_penalty=0.0004, weight=1.0000

### y3

- AUROC (macro): 0.4842; AUROC (micro): 0.4842
- AUPRC (macro): 0.5012; AUPRC (micro): 0.5012
- Accuracy: 0.5033
- F1 macro: 0.6519
- Losses: total=2.1237, recon=1.4273, KL=0.0085, CE=0.6946, info_penalty=0.0000, weight=1.0000
