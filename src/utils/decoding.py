import torch.nn as nn


def get_decoding_strategy(model: nn.Module) -> str:
    """
    Factory logic to determine decoding strategy.
    """
    # Check for Stepwise signature (e.g. from your StepwiseScenarioReducer)
    if hasattr(model, "decoder") and hasattr(model.decoder, "_greedy"):
        return "stepwise"
    # Fallback / Explicit One-Shot
    return "one_shot"
