# Small-Case Verification (n = 3, 4)

|n|seed|method|success|status|runtime_seconds|objective_value|marginal_violation|iterations|solver_message|simplex_objective|objective_gap_vs_simplex|
|---|---|---|---|---|---|---|---|---|---|---|---|
|3|11|simplex|True|optimal|0.004587|10.10746260|0.00e+00|6|Optimization terminated successfully. (HiGHS Status 7: Optimal)|10.107462599223187|0.0|
|3|11|ipm|True|optimal|0.001193|10.10746260|0.00e+00|7|Optimization terminated successfully. (HiGHS Status 7: Optimal)|10.107462599223187|0.0|
|3|11|pdlp|True|optimal|0.000195|10.10746260|7.76e-12|64|Solved with OR-Tools PDLP.|10.107462599223187|-3.070255161219393e-11|
|4|17|simplex|True|optimal|0.000616|12.49922026|0.00e+00|6|Optimization terminated successfully. (HiGHS Status 7: Optimal)|12.499220260651615|0.0|
|4|17|ipm|True|optimal|0.000614|12.49922026|0.00e+00|8|Optimization terminated successfully. (HiGHS Status 7: Optimal)|12.499220260651615|0.0|
|4|17|pdlp|True|optimal|0.000143|12.49922026|2.89e-14|128|Solved with OR-Tools PDLP.|12.499220260651615|-1.2434497875801753e-13|
