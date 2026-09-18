# Differential Privacy on Banking data
## Prior Works
I used [DP-SGD](https://github.com/kartikey2807/Differentially-Private-SGD) as the starting point for this exercise. Refer to: [Abadi et. al.](https://arxiv.org/pdf/1607.00133)

*[details from past project]*

- Used DP-SGD to train a cnn on the MNIST dataset
- Compared validation accuracy against Vanilla SGD
- To hide each record:
    - Clip its contribution (gradients)
    - Bound the L2-sensitivity
    - Inject Gaussian noise to the aggregate, which ensures $(\epsilon,\delta)$-differential privacy
    - Why not Laplacian noise?
        - In the case of Gaussian, you inject less noise for the same $\epsilon$
        - Because Gaussian uses *L2-sensitivity*
    - To calibrate $\delta$:
        - Follow the heuristic --> $\delta << \frac{1}{N}$
        - Where, $N$ is the population size
        - You want a low failure probability relative to population size
    - To calibrate $\epsilon$:
        - Ideally, you first decide what $\epsilon$ you prefer
        - and then tune the noise around it
        - $\sigma_n = \frac{\Delta_2 \times \sqrt{2\ln{(1.25/\delta)}}}{\epsilon} = \frac{C}{B} \times \frac{\sqrt{2\ln{(1.25/\delta)}}}{\epsilon}$
        - where $\Delta_2$ is the *L2-sensitivity* and $\Delta_2 = \frac{C}{B}$, the clipping parameter. Second term is the noise multiplier.

- There are ***two*** key limitations:
    - Iterative noise injection cause performance degradation wrt time
    - Has a $\delta$ probability of complete privacy failure, though $\delta$ is small and privacy tends to degrade gracefully.

Other resources: [robust FL](https://github.com/kartikey2807/robust-federated-learning)

- This project focuses on federated learning
- But, here too, gradient transfer exists between clients and the server
- An attacker can:
   - capture these gradients
   - reconstruct the original client dataset *threat*
- I took inspiration from the DP-GAN paper by [Xie et. al.](https://arxiv.org/pdf/1802.06739) to clip and inject Gaussian noise into shared gradients.

## Terminology

***What is a privacy unit? (or protected contribution)***

- This is the data point you want to safeguard
- For this exercise, it could be
  - One Subject, or
  - One transaction

***How do you define neighboring datasets?***

- Two dataset $D$ and $\hat{D}$ are neighboring if they differ by one data point
- In this case, they could be of two types:
   - *transaction-level neighbors*
      - One transaction is added or removed between datasets.
   - *subject-level neighbors*
      - One subject is added or removed, but it also leads to
        - Associated accounts being added/removed
        - Associated transactions being added/removed
        - Changes the edge and node features for other Subjects

***For our implementation***

- We consider each transaction as a protected data point and hence assume a transaction-level neighborhood.

## Dataset

Dataset was generated using *generate_transactions.py*.
- 8 subjects
- 11 accounts
- 2 banks
- 50 transactions
- separate bank A and B views
- repeated transactions
- subjects with multiple accounts in various banks
- payment chain between banks

*Deterministic dataset*

|Subject|Bank Group|Accounts|Banks|
|:------|:---------|:-------|:----|
|S1|A-primary|A1, A2|Bank A|
|S2|A-primary, cross-bank|A3, B1|Banks A and B|
|S3|A-primary, cross-bank|A4, B2|Banks A and B|
|S4|A-primary|A5|Bank A|
|S5|A-primary|A6|Bank A|
|S6|A-primary|A7|Bank A|
|S7|B-primary|B3|Bank B|
|S8|B-primary|B4|Bank B|

Transactions fields
```
transaction_id
amount
sender_id
recipient_id
sender_account
recipient_account
sender_bank
recipient_bank
timestamp
```
*Subject 2 owns multiple accounts, and dropping it leads to deleting 26 transactions and affecting other subjects as well.*
*Payment chain*: `S1:A1 → S2:A3 → S3:B2 → S7:B3 → S3:A4 → S2:B1 → S4:A5`

Node features
```
outgoing_count
incoming_count
total_sent
total_received
```

Edge features
```
source account
target account
amount
timestamp
sender bank
recipient bank
sender subject
recipient subject
is_cross_bank
```

## Model

I used a 2-parameter tensors: one weight and one bias term. The Exercise.md mentions no full GNN implementation.

```python
class TwoTensorModel(nn.Module):
    """A model with exactly two parameter tensors: weight and bias."""

    def __init__(self, input_dim: int, output_dim: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)
```

## Methodology

```python
## Hyper-parameters
epsilon = 1
delta = 0.01
```

- Transfer the gradient update from my DPSGD project
  - Suppose we get gradients $g_i$
    - $g_i = \[\nabla_W \text{ Loss}, \nabla_b \text{ Loss}\]$
  - Clip gradient $g_i$ to obtain $\hat{g_i}$
    - $\hat{g_i} = g_i \times \min\(1,\frac{C}{\lVert g_i \rVert_2}\)$
  - Compute average gradients
    - $f(D) = \frac{1}{B} \sum_{i \in D} \hat{g_i}$
  - Here you compute sensitivity as well
    - $\Delta_2 = \lVert f(D) - f(\hat{D}) \rVert_2 \leq \frac{C}{B}$
    - Bounded by $\frac{C}{B}$ because the norms were clipped (see the assumption below)
  - $B = 50$ *all samples in 1 batch*
  - Final noisy gradient update is
    - $g(D) = \frac{1}{B} \( \sum_{i \in D} \hat{g_i} + \mathcal{N}(0,\sigma^2 C^2)\) \implies \frac{1}{B} \sum_{i \in D} \hat{g_i} + \mathcal{N}(0,\frac{\sigma^2 C^2}{B^2})$
    - And again we see that $\sigma_n = \frac{\sigma C}{B}$
    - Noise scales due to sensitivity
- Claim privacy for transaction-level data
- Measure how much the gradient changes with $1$ transaction removed
- Does it breach the privacy bounds?
- Extend this to subject-level neighborhood and see if the bound holds
- If it does not, reason as to why it failed. I have an assumption that it will fail for subject with multiple accounts.

***How do you calibrate the clipping parameter?***

Thumb rule:-
- Capture all unclipped gradients
- From their distribution, set clip to the 90th percentile of the distribution.
- `C = 1.2584635019`

## Sensitivity Assumption (Denominator)

For the nominal transaction-level sensitivity calculations, we assume that removing one transaction would not affect other transactions' features and gradients. Under this **frozen-feature assumption**, with per-transaction grad clipped to norm *C* and an average over a fixed denominator *B*, we get $\lVert f(D) - f(\hat{D})\rVert_2 \leq \frac{C}{B}$. Here $D$ is obtained by removing or adding one transaction record to $\hat{D}$. In implementation, node features are rebuilt after a subject is deleted. Hence, removing or adding a transaction can change node features for the sender and the recipient. So $\frac{C}{B}$ should be treated as a **reference-sensitivity bound**, not as the empirical bound for our use case.

***How to run the scripts?***

From the project directory
```bash
source venv/Scripts/activate
python main.py --mode transaction --transaction-id T001 ## Run the transaction-level check
python main.py --mode subject --subject-id S1 ## Run the subject-level check
```

***Examples***
```bash
python .\main.py --mode transaction --transaction-id T021
```
```
Privacy sensitivity check
==========================
Mode: transaction
Baseline transactions: 50
Transactions removed: 1
Clipping bound C (90th percentile): 1.2584635019
Fixed denominator B: 50

Result
------
Measured pre-noise gradient difference: 0.0171917379
Expected transaction bound C/B:         0.0251692700
Check passed: True

Interpretation
--------------
This is the formal transaction-level add/remove sensitivity check.
Noise is constructed for the released update, while the reported difference is a deterministic pre-noise measurement.
```

```bash
python .\main.py --mode subject --subject-id S2
```
```
Privacy sensitivity check
==========================
Mode: subject
Baseline transactions: 50
Transactions removed: 26
Clipping bound C (90th percentile): 1.2584635019
Fixed denominator B: 50

Result
------
Measured pre-noise gradient difference: 0.1150961816
Transaction-level reference bound C/B:  0.0251692700
Transaction bound holds in this subject test: False
Subject-level privacy bound claimed: False

Interpretation
--------------
This is a subject-removal stress test. The mechanism remains transaction-level, so no subject-level privacy bound is claimed.
Noise is constructed for the released update, while the reported difference is a deterministic pre-noise measurement.
```

## Observation

When a transaction is removed:
- The change in updates does not exceed the claimed bound $\frac{C}{B} = 0.02517$.

When a subject is dropped:
- all associated accounts are dropped
- all the adjacent transactions are dropped
- The aggregated update change exceeds the bound every time.

*Due to time constraints, I couldn't write about privacy accounting, cross-bank graphs, and the exponential mechanism.*   
*We could discuss that on call.*

## Use of AI

*I used the GPT-5.6 Agent to write scripts to generate transactions, build/re-build node and edge features, and compute privacy bounds and gradient updates. I did go through the privacy bound and gradient script myself in the 2-hour time.*
