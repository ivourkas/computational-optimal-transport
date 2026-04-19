# Optional PDHG vs PDLP Comparison

|n|seed|solver|success|status|runtime_seconds|objective_value|marginal_violation|iterations|pdlp_objective|objective_gap_vs_pdlp|
|---|---|---|---|---|---|---|---|---|---|---|
|50|1050|pdhg_numpy|True|converged|0.252370|12.61027330|9.86e-07|12104|12.610275092106235|-1.79e-06|
|50|1050|pdlp|True|optimal|0.021529|12.61027509|1.30e-14|1280|12.610275092106235|+0.00e+00|
|100|1100|pdhg_numpy|True|converged|1.293842|10.82065685|9.97e-07|29526|10.820657998920092|-1.15e-06|
|100|1100|pdlp|True|optimal|0.146535|10.82065800|1.21e-14|2496|10.820657998920092|+0.00e+00|
|200|1200|pdhg_numpy|False|max_iterations|3.491670|8.56303399|2.11e-01|30000|11.010308713620166|-2.45e+00|
|200|1200|pdlp|True|optimal|0.593684|11.01030871|2.68e-13|2624|11.010308713620166|+0.00e+00|
|500|1500|pdhg_numpy|False|max_iterations|18.190209|1.18420039|1.13e+00|30000|10.389507989560876|-9.21e+00|
|500|1500|pdlp|True|optimal|5.376949|10.38950799|8.19e-09|3328|10.389507989560876|+0.00e+00|
