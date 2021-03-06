from CNN import *

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a CNN for saliency prediction",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.03, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=1000,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--lr-decay",
    default=True,
    type=bool,
    help="Enable/disable learning decay",
)
parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
    help="Enable/disable momentum",
)
parser.add_argument(
    "--weight-decay",
    default=0.0005,
    type=float,
    help="Enable/disable weight decay",
)
parser.add_argument(
    "--nesterov",
    default=True,
    type=bool,
    help="Enable/disable nesterov",
)

### CHECKPOINT ###
parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoint"))
parser.add_argument("--checkpoint-frequency", type=int, default=1, help="Save a checkpoint every N epochs")
parser.add_argument("--resume-checkpoint", type=Path)


def main(args):
    model = CNN(height=96, width=96, channels=3)

    ### CHECKPOINT - load parameters, args, loss ###
    if args.resume_checkpoint != None and args.resume_checkpoint.exists():
        checkpoint = torch.load(args.resume_checkpoint)
        print(f"Resuming model {args.resume_checkpoint} that achieved {checkpoint['loss']} loss")
        model.load_state_dict(checkpoint['model'])
        old_epochs = args.epochs
        args = checkpoint['args']
        args.epochs -= old_epochs


    train_loader = torch.utils.data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    criterion = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
        args=args
    )

    print("done training")

    summary_writer.close()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        args,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()

        # learning rate decay
        lrs = np.linspace(0.03, 0.0001, epochs) if args.lr_decay else np.ones(epochs) * args.learning_rate

        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()

            if args.lr_decay:
                for g in self.optimizer.param_groups:
                    g['lr'] = lrs[epoch]

            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                logits = self.model.forward(batch)

                loss = self.criterion(logits, labels)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, loss, data_load_time, step_time, lrs[epoch])

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)

            # ### CHECKPOINT - save parameters, args, accuracy ###
            # Save every args.checkpoint_frequency or if this is the last epoch
            if (epoch + 1) % args.checkpoint_frequency == 0 or (epoch + 1) == epochs:
                print(f"Saving model to {args.checkpoint_path}")
                torch.save({
                    'args': args,
                    'model': self.model.state_dict(),
                    'loss': loss.item()
                }, args.checkpoint_path)

            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def validate(self):
        preds = []
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds.append(logits.cpu().numpy())

        average_loss = total_loss / len(self.val_loader)

        print(f"validation loss: {average_loss:.5f}")

        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )

    def print_metrics(self, epoch, loss, data_load_time, step_time, lr):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"lr: {lr:.5f}, "
                f"batch loss: {loss:.5f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
          f"CNN_"
          f"bs={args.batch_size}_" +
          f"lr={args.learning_rate}_" +
          ("lr-decay_" if args.lr_decay else "") +
          f"weight-decay={args.weight_decay}_" +
          f"momentum={args.momentum}_" +
          ("nesterov_" if args.nesterov else "") +
          f"run_"
      )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    main(parser.parse_args())
