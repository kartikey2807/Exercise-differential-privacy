# Node16 privacy reasoning exercise

This is a voluntary, unpaid candidate exercise. Spend at most two hours,
including reading and coding, then stop and report what remains unresolved.
We suggest a 30–45 minute discussion afterwards. You may instead bring existing
work that demonstrates the same skills. AI tools are allowed; explain which
conclusions and code you independently checked.

## Context

Node16 develops financial-crime detection across banks. Each bank keeps its own
records and lawful customer/payment information. Banks receive shared model
updates. We need to distinguish learning useful patterns from revealing another
bank's previously unseen subject-specific information. One subject may own
several accounts at several banks. Removing that subject can remove transactions
and change the graph features of other subjects.

Three capabilities have separate information boundaries: a shared detection
model, case investigation with controlled disclosure, and authorized bank-to-bank
sharing. This exercise concerns the model and its audit; it does not ask for a
legal opinion or the design of our complete privacy architecture.

## Starting from your work

Your [DP-SGD notebook](https://github.com/kartikey2807/Differentially-Private-SGD)
and [robust federated-learning project](https://github.com/kartikey2807/robust-federated-learning)
are relevant starting points. Choose the update path you can explain best.
Briefly identify your contribution and any code or method you built on.

Your supervisor's [ExpM+NF repository that you shared](https://github.com/bridgesra/expm_nf_mimic3_results_code)
is useful background for the discussion. You do not need to debug that project
or reproduce its MIMIC results. Use a revision of your own work that you know
and record its commit or link so we can discuss the same implementation.

## Practical task

Prioritize the executable clipping check and your explanation of its limitations.
A correct, clearly explained partial result is sufficient. Privacy accounting
and cross-bank graph effects may be discussed briefly. The paper comparison
below is optional preparation for our conversation.

Take one clipping/noise update from your chosen project. State what one
protected contribution is, what is clipped, how noise is calibrated and what
the receiver observes. Explain how you would determine epsilon and delta over
the released training sequence, including any assumptions or missing accounting.
You do not need to calculate a certified budget for Node16.

Create or reuse a tiny synthetic transaction example to make the check concrete.
A few subjects, two banks and roughly 20–100 transactions are enough; use fewer
if they demonstrate the issue. Include account-to-subject ownership, sender and
recipient accounts, bank ownership, timestamps and amounts. Give one subject
multiple accounts and repeated transactions, and include a short payment chain
crossing the two banks. Keep each bank's view separate. Describe any invented
ownership mappings or labels. No real customer information is needed.

A short Python generator or hand-written table is sufficient. You may use a
small sample from [AMLSim](https://github.com/IBM/AMLSim) or another synthetic
source if already available; cite its source and your changes. Do not spend the
exercise installing a simulator or generating a large dataset. The example
illustrates privacy assumptions; it need not establish detection quality.

Use this example for one executable check of your chosen clipping/noise logic.
A tiny model with a single update is sufficient. Alternatively, assign explicit
toy gradients to transactions and explain which model or graph effects this
simplification leaves out. Use at least two parameter tensors. State the bound
you claim, the norm and the neighboring-dataset definition it uses. Then compare
the update before noise when one transaction is removed and when all accounts
and incident transactions of one subject are removed. Make the handling of sums,
averages and the denominator explicit. If your implementation only supports a
transaction-level bound, explain why it does or does not extend to a subject
with multiple transactions.

Check whether the observed changes agree with your bound. If you find a
mismatch, show the smallest correction and rerun the check. Otherwise give a
conditional explanation and state what the check cannot prove. Explain how
removing the subject could change other subjects' graph features, and how a
subject appearing at both banks affects the privacy argument. Rebuild any
transaction-derived features you actually use after deletion. No full GNN or
federated training campaign is required.

## Optional discussion preparation

We will discuss the following questions together; no written answer is required.
If time remains within the two-hour limit, you may read the abstract, protocol
overview and limitations of FedGraphHE:
https://doi.org/10.1371/journal.pone.0339881

For the setting above, where might homomorphic encryption help? Who would hold
keys and see decrypted results? What could the receiving bank still infer from
the released model? Relate this to the DP path you checked and the exponential-mechanism work
you shared. Which protection, if any, would you investigate first for the
shared model, and what assumption or measurement would decide it?
No HE implementation, literature survey or overall architecture proposal is
expected. You do not need to read the paper or prepare the comparison in advance.

## Deliverable

One script with its tiny data or data generator, its exact run command and
output, plus at most one page of findings.

Include dependencies, assumptions, AI use and unfinished work. Use your own
machine; no GPU, paid cloud service, private repository or real data is needed.
A failed check or incomplete result is acceptable if you show what you attempted, what the result supports, the specific unresolved issue, and the next check you would run. 
A general statement that evidence is insufficient is not enough.
Do not spend more than the two-hour limit to make the report, and no need to make it look polished.

During the discussion we may change one assumption, such as adding a
transaction-derived feature or allowing an attacker to see every model round.
We may also ask what an unsuccessful membership attack establishes. Predict
what changes before trying it.
