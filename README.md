# Credit Card Customer Default Analysis

This project analyzes customer balances, payments, credit utilization, and repayment status, then predicts each test customer's probability of default.

The target is binary (`default`). The script compares logistic regression with an ordinary linear probability regression, then selects the linear model because it performs better on the supplied holdout and produces well-calibrated probabilities here. Its predictions are clipped to the valid probability range `[0, 1]`.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 credit_default_model.py \
  --data-dir /path/to/inter-uni-datathon-stream-1-credit-card-clients \
  --output submission.csv
```

The script prints validation metrics and a status summary grouped by default outcome. The output has the required columns:

```text
client_id,default_probability
```