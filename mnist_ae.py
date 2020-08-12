from torch.utils.data import DataLoader
import torchvision as tv
from tqdm import tqdm
from torch import nn, optim, save, load
import os
import torch


class ConvAE_1x28x28(nn.Module):

    def __init__(self, n_channels):
        super(ConvAE_1x28x28, self).__init__()

        self.encoder = nn.ModuleList()
        self.encoder.extend(self.add_encoder_block(n_channels, 64))
        # self.encoder.extend(self.add_encoder_block(64, 128))
        # self.encoder.extend(self.add_encoder_block(128, 256))

        self.decoder = nn.ModuleList()
        # self.decoder.extend(self.add_decoder_block(256, 128))
        # self.decoder.extend(self.add_decoder_block(128, 64))
        # self.decoder.extend(self.add_decoder_block(64, n_channels))

    def forward(self, x, verbose=False):
        if verbose:
            print(f"Input: {x.shape}")

        # encoding
        for e in self.encoder:
            x = e(x)
            if verbose:
                print(f"{type(e).__name__}: {x.shape}")

        # decoding
        for d in self.decoder:
            x = d(x)
            if verbose:
                print(f"{type(d).__name__}: {x.shape}")

        if verbose:
            print(f"Output: {x.shape}")
        return x


    @staticmethod
    def add_encoder_block(in_channels, out_channels,
                          kernel_size=3, relu_factor=0.0):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=3, padding=1),
            nn.LeakyReLU(relu_factor, inplace=True),
            nn.MaxPool2d(2, stride=2)
        ]

    @staticmethod
    def add_decoder_block(in_channels, out_channels,
                          kernel_size=3, relu_factor=0.0):
        return [
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=3),
            # nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(relu_factor, inplace=True)
        ]


class LinearAE_28x28(nn.Module):
    def __init__(self):
        super(LinearAE_28x28, self).__init__()

        self.encoder = nn.ModuleList()
        self.encoder.extend([
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        ])

        self.decoder = nn.ModuleList()
        self.decoder.extend([
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 28*28), nn.Tanh()
        ])

    def forward(self, x, verbose=False):

        if verbose:
            print(f"Input: {x.shape}")

        y = x.view(x.shape[0], -1)

        for e in self.encoder:
            y = e(y)
            if verbose:
                print(f"{type(e).__name__}: {y.shape}")

        for d in self.decoder:
            y = d(y)
            if verbose:
                print(f"{type(d).__name__}: {y.shape}")

        z = y.view_as(x)
        if verbose:
            print(f"Output: {z.shape}")

        return z

class MNIST_AE():
    def __init__(self, data_dir, output_dir, batch_size=16, device='cpu'):
        self.device = device

        # setup output_dir
        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Create Loader
        mnist_data = tv.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5], [0.5])
            ])
        )

        self.train_gen = DataLoader(
            mnist_data,
            batch_size=batch_size,
            shuffle=True
        )

        # Create encoder
        self.model = LinearAE_28x28().to(self.device)

        # Define optimizer & loss functions
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.loss = nn.MSELoss()

        # INTERNAL VARIABLES
        self.__hyperparam_fname = os.path.join(self.output_dir, f"{self.model.__class__.__name__}_hyperparams.pth")
        self.__chkpt_fname = os.path.join(self.output_dir, f"{self.model.__class__.__name__}_CHKPT.pth")

    def get_sample(self):
        return next(iter(self.train_gen))

    def print_model_output(self):
        img, _ = self.get_sample()
        img = img.to(self.device)
        self.model(img, verbose=True)

    def print_state(self):
        print(f"Status of {len(self.model.state_dict())} parameters:")
        for param in self.model.state_dict():
            print(f"  {param}: {self.model.state_dict()[param].shape}")

        print(f"Status of {len(self.optimizer.state_dict())} optimizer variables:")
        for var_name in self.optimizer.state_dict():
            print(f"  {var_name}: {self.optimizer.state_dict()[var_name]}")

    def train(self, num_epoch, model_from=None):
        """
        Train the model from the beginning.
        Use the model_from="path/to/model/file.pth" argument to load a model from the file.
        """

        assert num_epoch > 0, "num_epoch must be positive"

        # load a model
        if model_from is not None:
            self.model.load_state_dict(load(model_from))
            self.model.eval()

        # save the hyperparameters
        save({"num_epoch": num_epoch, "optimizer_stat_dict": self.optimizer.state_dict(), "loss": self.loss.state_dict()},
             self.__hyperparam_fname)

        # Start training
        for epoch in range(num_epoch):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        """
        Perform 1 epoch training. No model / loader validity check.
        """

        # Set model to training
        self.model.train()

        # Create a progress bar
        with tqdm(total=len(self.train_gen)) as pbar:

            # setup the progress bar
            pbar.set_description(f"Epoch {epoch}")
            avg_loss = 0.0

            # Run for all batches
            for batch_idx, (img_in, _) in enumerate(self.train_gen):

                # match the device
                img_in = img_in.to(self.device)

                # The 5 holy statements of torch's training:
                # -------------------------------------------
                self.optimizer.zero_grad()         # reset gradient calculation
                img_out = self.model(img_in)       # predict
                loss = self.loss(img_out, img_in)  # we compare image out == image in
                loss.backward()                    # back propagation
                self.optimizer.step()              # step forward
                # -------------------------------------------

                # update epoch loss
                avg_loss += loss.item()

                # Update tqdm progress bar
                pbar.set_postfix(
                    batch_loss=f"{loss.item():.2f}",
                    avg_loss=f"{avg_loss / (batch_idx+1):.2f}"
                )
                pbar.update()

        # final epoch_loss
        avg_loss /= len(self.train_gen)

        # After each epoch, we save the model
        save_fname = os.path.join(self.output_dir, f"{self.model.__class__.__name__}_EPOCH{epoch}.pth")
        save(self.model.state_dict(), save_fname)

        # And save the checkpoint to resume
        save({
            "epoch": epoch,
            "last_model_state": save_fname,
            "hyperparams_state": self.__hyperparam_fname
        }, self.__chkpt_fname)

        # return epoch result
        return {"loss": avg_loss}

    def random_test(self):
        """
        Randomly pick a batch sample from the training data and generate its AE results.

        Returns the following dictionary:
          image_in:  BATCH_SIZE x 1 x 28 x 28 of the randomly selected batch training data
          image_out: BATCH_SIZE x 1 x 28 x 28 of the generated auto-encoder from image_in
          label:     BATCH_SIZE array of the number label (0-9)
          loss:      the average loss value for this batch
        """

        img_in, label = self.get_sample()
        img_in = img_in.to(self.device)
        img_out = self.model(img_in)
        loss = self.loss(img_out, img_in)

        return {
            "image_in": img_in,
            "image_out": img_out,
            "label": label,
            "loss": loss.item()
        }

    def load_last_model(self):
        """
        Load the last checkpoint model

        Returns dictionary:
          epoch: last epoch number
          num_epoch: number of maximum epoch
        """

        assert os.path.isfile(self.__chkpt_fname), f"Cannot find the last check point model: {self.__chkpt_fname}"
        chkpt = load(self.__chkpt_fname)
        self.model.load_state_dict(load(chkpt['last_model_state']))
        hyperparams = load(chkpt['hyperparams_state'])
        self.optimizer.load_state_dict(hyperparams['optimizer_stat_dict'])
        self.loss.load_state_dict(hyperparams['loss'])

        return {'epoch': chkpt['epoch'], 'num_epoch': hyperparams['num_epoch']}

    def eval_encoder(self, x):
        """
        Evaluate the encoder part of the model
        """
        self.model.eval()

        y = x.to(self.device).view(x.shape[0], -1)
        for enc in self.model.encoder:
            y = enc(y)

        return y

    def eval_decoder(self, x):
        """
        Evaluate the decoder part of the model
        """
        self.model.eval()

        y = x.to(self.device)
        for dec in self.model.decoder:
            y = dec(y)

        return y

