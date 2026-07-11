# FHVAC
Link: https://doi.org/10.1109/TPDS.2024.3372046

Before solving with the Rule-Based scheme, it is necessary to fit the parameters in the two models with real data, namely run `constancs_fit.py`. Then run `Rule_Based.py` test plan.

`IL.by` is the neural network structure for converting Rule-Based schemes, and run `IL_train` to start training.

`PPO.py` and `PPO_train.py` are similar to CASVA.

`Fusion.py` is a network structure that combines two schemes, and run `fusion_train.py` to start training.

run `python main.py` to test.

