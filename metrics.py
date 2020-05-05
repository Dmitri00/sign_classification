from ignite.metrics import Accuracy, Loss, Recall
def get_metrics()
    metrics={
        'accuracy': Accuracy(),
        'nll': Loss(criterion),
        'recall': Recall(),
        }
    return metrics