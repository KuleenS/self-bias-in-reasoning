"""selfbias — measuring self-bias in LLM reasoning.

The headline experiment fits

    score_ijkl = b0 + b_self * I_self
                 + u_generator_i + u_evaluator_j + u_dataset_k + u_prompt_l + eps

where `score` is whether evaluator j judges generator i's reasoning chain on prompt l of
dataset k as correct, and `I_self` is 1 when the generator and evaluator are the same model.
"""

__version__ = "0.1.0"
