import torch
import os
import datetime
from tensorboardX import SummaryWriter

class Logger:

    SUM_FREQ = 10

    def __init__(self, args, scheduler):
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.name = args.name
        self.scheduler = scheduler

        self.path = os.path.join("train_log", args.name)
        self.checkpoints_folder = os.path.join(self.path, "checkpoints")
        self.SummaryWriter_folder = os.path.join(self.path, "SummaryWriter")
        self.log_folder = os.path.join(self.path, "train_log.txt")

        if not os.path.exists(self.path):
            os.makedirs(self.checkpoints_folder)
            os.makedirs(self.SummaryWriter_folder)

    def _print_training_status(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.SummaryWriter_folder)

        lr = self.scheduler.get_last_lr().pop()
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        training_str = "[{}, {:6d}, {:7.5f}] ".format(now_time, self.total_steps+1, lr)
        metrics_data = [self.running_loss[k] / self.SUM_FREQ for k in self.running_loss.keys()]
        metrics_str = ""
        for k in self.running_loss.keys():
            value = self.running_loss[k] / self.SUM_FREQ
            metrics_str += f"{k}={value:6.4f}, "
        # metrics_str = ("{}:{:6.4f}, "*len(metrics_data)).format(*self.running_loss.keys(), *metrics_data)
        
        # print the training status
        self._log(training_str + metrics_str)

        for key in self.running_loss:
            val = self.running_loss[key] / self.SUM_FREQ
            self.writer.add_scalar(key, val, self.total_steps)
            self.running_loss[key] = 0.0

    def _log(self, string):
        with open(self.log_folder, "a") as f:
            f.write(string + "\n")
        print(string)

    def push(self, metrics):
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.SUM_FREQ == self.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

    def checkpoints(self, model, total_steps):
        cks_path = os.path.join(self.checkpoints_folder, f"{total_steps:0>6d}.pth")
        torch.save(model.state_dict(), cks_path)

    def write_dict(self, results):
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

