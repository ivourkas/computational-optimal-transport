# Part 3(a,b): Sinkhorn and Conic Comparison

|n|seed|lambda|method|success|status|runtime_seconds|objective_value|transport_cost|marginal_violation|iterations|
|---|---|---|---|---|---|---|---|---|---|---|
|200|1200|1|ipm|True|optimal|1.414634|2.09189032|11.79680269|6.89e-10|49|
|200|1200|1|sinkhorn|True|converged|0.015431|2.09188569|11.79679815|8.22e-07|30|
|200|1200|1|sinkhorn_numpy|True|converged|0.018752|2.09189021|11.79680264|1.25e-08|40|
|200|1200|0.1|ipm|True|optimal|1.262731|10.29683780|11.07891118|4.35e-09|42|
|200|1200|0.1|sinkhorn|True|converged|0.178356|10.29681531|11.07888851|4.10e-06|360|
|200|1200|0.1|sinkhorn_numpy|True|converged|0.211281|10.29683102|11.07890399|9.76e-07|420|
|200|1200|0.01|ipm|True|optimal|1.105168|10.95057873|11.01363984|1.80e-08|35|
|200|1200|0.01|sinkhorn|True|converged|10.949355|10.95055151|11.01361285|4.59e-06|20730|
|200|1200|0.01|sinkhorn_numpy|False|max_iterations|26.120623|10.95056762|11.01362877|1.79e-06|50000|
