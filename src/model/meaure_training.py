import psutil
import time
import pandas as pd
from pytorch_lightning import Callback

class CPUUsageCallback(Callback):
    def __init__(self,output, what="epochs"):
        self.batch_size = 5
        self.process = psutil.Process()
        self.cpu_start = None
        self.start = None
        self.Output = output
        self.cpu_end = None
        self.end = None
        self.latency = None
        self.cpu_usage = None
        self.what = what
        self.OutCPU = pd.DataFrame(columns=["Step Epoch", "latency min", "cpu_usage %"])
        self.state = {"epochs": 0}
       
    def on_train_start(self, *args, **kwargs):
        self.cpu_start = self.process.cpu_percent()
        self.start = time.time()
        

    def on_train_epoch_end(self, trainer,*args, **kwargs ):
#         epoch = trainer.current_epoch
        if self.what == "epochs":
            self.state["epochs"] += 1

        self.end = time.time()
        self.cpu_end = self.process.cpu_percent()
        self.latency = self.end - self.start
        self.cpu_usage = self.cpu_end - self.cpu_start
        # Append the latency value to the DataFrame
        self.OutCPU = self.OutCPU.append({"Step Epoch": self.state["epochs"], "latency min": self.latency, "cpu_usage %": self.cpu_usage}, ignore_index=True)

   
    def on_train_end(self, *args, **kwargs):
        self.OutCPU.to_csv(self.Output+"cpu_metrics.csv", index=False)

class ThroughputCallback(Callback):
    def __init__(self, output,what="batches"):
        super(ThroughputCallback, self).__init__()
        self.what = what
        self.start = None
        self.Output = output
        self.batch_size = 5
        self.state = {"epochs": 0, "batches": 0}
        self.Throughput = pd.DataFrame(columns=["epochs","batch_idx", "Throughput Image/Sec"])

    def on_train_start(self, trainer, pl_module):
        self.start = time.time()

    def on_after_backward(self, trainer, pl_module):
        batch_idx = trainer.global_step
        if self.what == "batches":
            self.state["batches"] += 1
            if batch_idx % self.batch_size == 0:
                end = time.time()
                latency = end - self.start
                throughput = self.batch_size / latency
                self.start = time.time()
                # Append the throughput value to the DataFrame
                self.Throughput = self.Throughput.append({"epochs":self.state["batches"],"batch_idx": batch_idx, "Throughput Image/Sec": throughput}, ignore_index=True)

    def on_train_end(self, trainer, pl_module):
        self.Throughput.to_csv(self.Output+"throughput_metrics.csv", index=False) 