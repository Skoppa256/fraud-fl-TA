# Per-client minority census (sampling_strategy = 0.01)

Sweep: 3 datasets x 4 schemes x 3 seeds x 5 clients = 180 client-instances.

Per dataset x scheme: aggregated over 15 client-instances (3 seeds x 5 clients).

| Dataset | Scheme | clients<10 | clients=0 | min | median | max | worst mult | fires | skip:insuff | skip:target |
|---------|--------|-----------|-----------|-----|--------|-----|-----------|-------|-------------|-------------|
| paysim | IID | 0 | 0 | 1069 | 1155 | 1205 | x7 | 15 | 0 | 0 |
| paysim | Dirichlet a=0.5 | 2 | 0 | 2 | 1080 | 2931 | x30 | 11 | 2 | 2 |
| paysim | Dirichlet a=1.0 | 0 | 0 | 88 | 1030 | 3158 | x123 | 13 | 0 | 2 |
| paysim | Dirichlet a=5.0 | 0 | 0 | 249 | 1164 | 1792 | x41 | 15 | 0 | 0 |
| creditcard | IID | 0 | 0 | 53 | 67 | 90 | x7 | 15 | 0 | 0 |
| creditcard | Dirichlet a=0.5 | 2 | 0 | 4 | 53 | 181 | x30 | 12 | 2 | 1 |
| creditcard | Dirichlet a=1.0 | 0 | 0 | 17 | 49 | 237 | x23 | 13 | 0 | 2 |
| creditcard | Dirichlet a=5.0 | 0 | 0 | 41 | 69 | 98 | x11 | 15 | 0 | 0 |
| baf | IID | 0 | 0 | 1494 | 1540 | 1620 | x0 | 0 | 0 | 15 |
| baf | Dirichlet a=0.5 | 1 | 0 | 3 | 1095 | 4976 | x185 | 6 | 1 | 8 |
| baf | Dirichlet a=1.0 | 0 | 0 | 128 | 1016 | 4693 | x13 | 7 | 0 | 8 |
| baf | Dirichlet a=5.0 | 0 | 0 | 782 | 1445 | 2291 | x1 | 7 | 0 | 8 |

## Claim 1 — most-exposed dataset (alpha=0.5)

- **paysim** a=0.5: 0 zero-minority, 2 <10-minority client-instances (of 15); min minority = 2.
- **creditcard** a=0.5: 0 zero-minority, 2 <10-minority client-instances (of 15); min minority = 4.
- **baf** a=0.5: 0 zero-minority, 1 <10-minority client-instances (of 15); min minority = 3.

## Claim 2 — `target_met` skips per dataset x scheme

- paysim IID: 0/15 skip via target_met.
- paysim Dirichlet a=0.5: 2/15 skip via target_met.
- paysim Dirichlet a=1.0: 2/15 skip via target_met.
- paysim Dirichlet a=5.0: 0/15 skip via target_met.
- creditcard IID: 0/15 skip via target_met.
- creditcard Dirichlet a=0.5: 1/15 skip via target_met.
- creditcard Dirichlet a=1.0: 2/15 skip via target_met.
- creditcard Dirichlet a=5.0: 0/15 skip via target_met.
- baf IID: 15/15 skip via target_met.
- baf Dirichlet a=0.5: 8/15 skip via target_met.
- baf Dirichlet a=1.0: 8/15 skip via target_met.
- baf Dirichlet a=5.0: 8/15 skip via target_met.

## Cells where SMOTE fires on ZERO clients (SMOTE arm == no-SMOTE arm)

- baf IID: 0/15 clients oversample.
