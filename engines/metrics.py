from ignite.metrics import Accuracy, Loss, Recall


def get_metrics(loss):
    metrics = {
        'accuracy': Accuracy(),
        'nll': Loss(loss),
        'recall': Recall(average=True),
    }
    return metrics
