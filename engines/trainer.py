from ignite.engine import Events, create_supervised_trainer
from string import Template


class Trainer:
    def __init__(self, model, loss, optimizer, lr_scheduler, device, logger, log_interval, output_dir=None):
        self.logger = logger
        self.lr_scheduler = lr_scheduler
        self.log_interval = log_interval
        self.progress_bar_desc = "ITERATION - loss: {:.2f}"
        self.trainer_engine = create_supervised_trainer(model, optimizer, loss, device=device)
        self.trainer_engine.add_event_handler(Events.ITERATION_COMPLETED(every=log_interval),
                                              self.log_training_loss)
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                              self.lr_step)

    def log_training_loss(self, engine):
        self.logger.pbar.desc = self.progress_bar_desc.format(engine.state.output)
        self.logger.pbar.update(self.log_interval)

    def lr_step(self, engine):
        self.lr_scheduler.step()

    def run(self, dataloader, max_epochs):
        self.trainer_engine.run(dataloader, max_epochs)
