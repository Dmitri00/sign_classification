from ignite.engine import Events, create_supervised_evaluator
from string import Template


class Evaluator:
    def __init__(self, trainer_engine, model, metrics, val_loader, logger):
        self.logger = logger
        self.val_loader = val_loader
        self.metrics = metrics
        self.val_results_fmt = self.build_val_results_fmt(metrics)
        self.evaluator_engine = create_supervised_evaluator(model, metrics)
        trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self.log_validation_results)

    def build_val_results_fmt(self, metrics):
        val_results_fmt = "Validation Results - Epoch: {epoch:}"
        for metric_name in metrics.keys():
            metric_value_fmt = Template('{$metric_name:$precision}').substitute(
                metric_name=metric_name, precision='5.2f'
            )
            metric_name_ = 'Avg ' + metric_name
            val_results_fmt += " {:>12}: {value_fmt:s}".format(metric_name_,
                                                               value_fmt=metric_value_fmt)
        return val_results_fmt

    def log_validation_results(self, engine):
        self.evaluator_engine.run(self.val_loader)
        metrics = self.evaluator_engine.state.metrics
        metric_values = {metric_name: metrics[metric_name] for metric_name in metrics}
        self.logger.tqdm.write(self.val_results_fmt
            .format(
            epoch=engine.state.epoch, **metric_values
        )
        )
        self.logger.pbar.n = self.logger.pbar.last_print_n = 0

    def run(self):
        self.evaluator_engine.run(self.val_loader)
