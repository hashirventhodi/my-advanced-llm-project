def compute_loss_metrics(metrics_dict, loss):
    """
    Optionally compute and store additional metrics, e.g. perplexity.
    """
    metrics_dict["loss"] = loss
    # For example:
    # metrics_dict["ppl"] = math.exp(loss)
    return metrics_dict
