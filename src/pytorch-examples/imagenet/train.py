import argparse
import os
import pathlib

import horovod.torch as hvd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data, tensorboard
from torchvision import datasets, transforms, models


parser = argparse.ArgumentParser(description="PyTorch + Horovod distributed training benchmark")
parser.add_argument("--data-dir",
                    type=str,
                    help="Path to ILSVR data")
parser.add_argument("--read-checkpoints-from",
                    type=str,
                    help="Path to a directory containing existing checkpoints")
parser.add_argument("--write-checkpoints-to",
                    type=str,
                    help="Path to the directory where checkpoints should be written")
parser.add_argument("--tensorboard-logging-dir",
                    type=str,
                    help="Path to the directory where tensorboard logs should be written")
parser.add_argument('--batches-per-allreduce',
                    type=int,
                    default=1,
                    help="number of batches processed locally before executing allreduce across workers")

# Most default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument("--batch-size",
                    type=int,
                    default=256,
                    help="input batch size for training")
parser.add_argument("--base-batch-size",
                    type=int,
                    default=32,
                    help="batch size used to determine number of effective GPUs")
parser.add_argument("--val-batch-size",
                    type=int,
                    default=32,
                    help="input batch size for validation")
parser.add_argument("--warmup-epochs",
                    type=float,
                    default=5,
                    help="number of warmup epochs")
parser.add_argument("--epochs",
                    type=int,
                    default=90,
                    help="number of epochs to train")
parser.add_argument("--base-lr",
                    type=float,
                    default=1.25e-2,
                    help="learning rate for a single GPU")
parser.add_argument("--max-lr",
                    type=float,
                    default=1e-2,
                    help="max learning rate for one-cycle scheduler")
parser.add_argument("--momentum",
                    type=float,
                    default=0.9,
                    help="SGD momentum")
parser.add_argument("--weight-decay",
                    type=float,
                    default=5e-5,
                    help="weight decay")
parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="random seed")
args = parser.parse_args()

# initialize horovod
hvd.init()
torch.manual_seed(args.seed)
torch.cuda.set_device(hvd.local_rank()) # Horovod: pin GPU to local rank.
torch.cuda.manual_seed(args.seed)

# create required directories
data_dir = pathlib.Path(args.data_dir)
training_dir = data_dir / "train"
validation_dir = data_dir / "val"
checkpoints_logging_dir = pathlib.Path(args.write_checkpoints_to)
tensorboard_logging_dir = pathlib.Path(args.tensorboard_logging_dir)
        
# define constants used in data preprocessing
resized_img_width, resized_img_height = 256, 256
target_img_width, target_img_height = 224, 224
n_training_images = 1281167
n_validation_images = 50000
n_testing_images = 100000


class Metric(object):
    
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

    
def _compute_accuracy(output, target):
    pred = output.max(1, keepdim=True)[1]
    accuracy = (pred.eq(target.view_as(pred))
                    .cpu()
                    .float()
                    .mean())
    return accuracy
    

def _partial_fit(model_fn, loss_fn, optimizer, loss_metric, accuracy_metric, X_batch, y_batch):
    
    # forward pass
    loss = loss_fn(model_fn(X_batch), y_batch)
    loss_metric.update(loss)
    accuracy_metric.update(_compute_accuracy(model_fn(X_batch), y_batch))

    # back propagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad() # don't forget to reset the gradient after each batch!


def _validate(model_fn, loss_fn, validation_data_loader, epoch, rank):
    
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with torch.no_grad():
        for X, y in validation_data_loader:
            val_loss.update(loss_fn(model_fn(X), y))
            val_accuracy.update(_compute_accuracy(model_fn(X), y))
        print(f"Training epoch: {epoch}, Validation loss: {val_loss.avg.item()}, Validation accuracy: {val_accuracy.avg.item()}")
    
    if rank == 0:
        summary_writer.add_scalar('val/loss', val_loss.avg, epoch)
        summary_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


def fit(model_fn, loss_fn, optimizer, lr_scheduler, training_data_loader, validation_data_loader, rank, initial_epoch, number_epochs):
    
    for epoch in range(initial_epoch, number_epochs):
        
        # train for a single epoch
        model_fn.train()
        train_sampler.set_epoch(epoch)
        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')
        
        for X_batch, y_batch in training_data_loader:
            _partial_fit(model_fn, loss_fn, optimizer, train_loss, train_accuracy, X_batch, y_batch)
            lr_scheduler.step()
        
        # compute validation loss after every epoch
        model_fn.eval()
        _validate(model_fn, loss_fn, validation_data_loader, epoch, rank)

        # only checkpoint on rank 0 worker to avoid corruption of checkpoint data
        if rank == 0:
            _checkpoint = {"model_state_dict": model_fn.state_dict(),
                           "optimizer_state_dict": optimizer.state_dict()}
            torch.save(_checkpoint, f"{checkpoints_logging_dir}/checkpoint-epoch-{epoch:02d}.pt")

            summary_writer.add_scalar('train/loss', train_loss.avg, epoch)
            summary_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)

# create training and validation data sets
_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder(data_dir / "train", transform=_train_transform)

_val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(data_dir / "val", transform=_val_transform)

# use DistributedSampler to partition data among workers
train_sampler = (data.distributed
                     .DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank()))
val_sampler = (data.distributed
                   .DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank()))

# create training and validation data loaders
class WrappedDataLoader:
    
    def __init__(self, data_loader, f):
        self._data_loader = data_loader
        self._f = f
        
    def __len__(self):
        return len(self._data_loader)
    
    def __iter__(self):
        for batch in iter(self._data_loader):
            yield self._f(*batch)

_data_loader_kwargs = {'num_workers': 6, "pin_memory": True}
_train_data_loader = (data.DataLoader(train_dataset,
                                      batch_size = args.batch_size * args.batches_per_allreduce,
                                      sampler=train_sampler,
                                      **_data_loader_kwargs))

_val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=args.val_batch_size,
                                               sampler=val_sampler,
                                               **_data_loader_kwargs)

_data_to_gpu = lambda X, y: (X.cuda(), y.cuda())
train_data_loader = WrappedDataLoader(_train_data_loader, _data_to_gpu)
val_data_loader = WrappedDataLoader(_val_data_loader, _data_to_gpu)

# set up standard ResNet-50 model.
model_fn = (models.resnet50()
                  .cuda())
loss_fn = F.cross_entropy
            
# adjust initial learning rate based on number of "effective GPUs".
_global_batch_size = args.batch_size * args.batches_per_allreduce * hvd.size()
_n_effective_gpus = _global_batch_size // args.base_batch_size 
_initial_lr = args.base_lr * _n_effective_gpus 
_optimizer = optim.SGD(model_fn.parameters(),
                       lr=_initial_lr,
                       momentum=args.momentum,
                       weight_decay=args.weight_decay)
distributed_optimizer = hvd.DistributedOptimizer(_optimizer,
                                                 named_parameters=model_fn.named_parameters(),
                                                 backward_passes_per_step=args.batches_per_allreduce)

# define learning rate scheduler
_lr_scheduler_kwargs = {
    "pct_start": 0.3,
    "anneal_strategy": "cos",
    "cycle_momentum": True,
    "base_momentum": 0.85,
    "max_momentum": 0.95,
    "div_factor": 25.0,
    "final_div_factor": 10000.0,
    "last_epoch": -1
}
one_cycle_lr = (torch.optim
                     .lr_scheduler
                     .OneCycleLR(distributed_optimizer,
                                 max_lr=args.max_lr,
                                 epochs=args.epochs,
                                 steps_per_epoch=n_training_images // (args.batch_size * hvd.size()),
                                 **_lr_scheduler_kwargs))

# only rank 0 worker should restore from checkpoint
_initial_epoch = 0
if hvd.rank() == 0:
    
    # Create the checkpoints and tensorboard logging directories (if necessary)
    if not os.path.isdir(checkpoints_logging_dir):
        os.mkdir(checkpoints_logging_dir)
    if not os.path.isdir(tensorboard_logging_dir):
        os.mkdir(tensorboard_logging_dir)
    
    # Look for a pre-existing checkpoint from which to resume training
    existing_checkpoints_dir = pathlib.Path(args.read_checkpoints_from)
    for _most_recent_epoch in range(args.epochs, 0, -1):
        _checkpoint_filepath = f"{existing_checkpoints_dir}/checkpoint-epoch-{_most_recent_epoch:02d}.pt"
        if os.path.exists(_checkpoint_filepath):
            _checkpoint = torch.load(_checkpoint_filepath)
            model_fn.load_state_dict(_checkpoint["model_state_dict"])
            distributed_optimizer.load_state_dict(_checkpoint["optimizer_state_dict"])
            _initial_epoch = _most_recent_epoch
            break
            
    summary_writer = tensorboard.SummaryWriter(tensorboard_logging_dir)
    
# broadcast initial epoch from rank 0 (which will have checkpoints) to other ranks.
initial_epoch = (hvd.broadcast(torch.tensor(_initial_epoch), root_rank=0)
                    .item())

# broadcast parameters & optimizer state from rank 0 to all other ranks
hvd.broadcast_parameters(model_fn.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(distributed_optimizer, root_rank=0)

# run the training loop
fit(model_fn, loss_fn, distributed_optimizer, one_cycle_lr, train_data_loader, val_data_loader, hvd.rank(), initial_epoch, args.epochs)




