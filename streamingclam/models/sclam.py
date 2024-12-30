import torch
from lightstream.modules.imagenet_template import ImageNetClassifier
from lightstream.models.resnet.resnet import split_resnet
from streamingclam.models.clam import CLAM_MB, CLAM_SB
from torchvision.models import resnet18, resnet34, resnet50
from torchmetrics.classification import Accuracy, AUROC

import os
import pathlib

# Streamingclam works with resnets, can be extended to other encoders if needed
class CLAMConfig(torch.nn.Module):
    def __init__(
        self,
        encoder: str,
        branch: str,
        n_classes: int = 2,
        gate: bool = True,
        use_dropout: bool = False,
        k_sample: int = 8,
        instance_loss_fn: torch.nn = torch.nn.CrossEntropyLoss,
        subtyping=False,
    ):
        self.branch = branch
        self.encoder = encoder
        self.n_classes = n_classes
        self.size = self.configure_size()
        
        self.gate = gate
        self.use_dropout = use_dropout
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping

    def configure_size(self):
        if self.encoder == "resnet50":
            return [2048, 512, 256]
        elif self.encoder == "resnet39":
            return [1024, 512, 256]
        elif self.encoder in ("resnet18", "resnet34"):
            return [512, 512, 256]

    def configure_clam(self):
        # size args original: self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        if self.branch == "sb":
            print("Loading CLAM with single branch \n")
            return CLAM_SB(
                gate=self.gate,
                size=self.size,
                dropout=self.use_dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                instance_loss_fn=self.instance_loss_fn(),
                subtyping=self.subtyping,
            )
        elif self.branch == "mb":
            print("Loading CLAM with multiple branches \n")
            return CLAM_MB(
                gate=self.gate,
                size=self.size,
                dropout=self.use_dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                instance_loss_fn=self.instance_loss_fn(),
                subtyping=self.subtyping,
            )
        else:
            raise NotImplementedError(
                f"branch must be specified as single-branch " f"'sb' or multi-branch 'mb', not {self.branch}"
            )


class StreamingCLAM(ImageNetClassifier):
    model_choices = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}

    def __init__(
        self,
        encoder: str,
        tile_size: int,
        loss_fn: torch.nn.functional,
        branch: str,
        n_classes: int,
        pooling_layer: str = "maxpool",
        pooling_kernel: int = 0,
        stream_pooling_kernel: bool = False,
        train_streaming_layers: bool = False,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
        unfreeze_at_epoch: int = 25,
        learning_rate: float = 2e-4,
        write_attention: bool = False,
        **kwargs,
    ):
        self.stream_pooling_kernel = stream_pooling_kernel
        self.pooling_layer = pooling_layer
        self.pooling_kernel = pooling_kernel
        self.instance_eval = instance_eval
        self.return_features = return_features
        self.attention_only = attention_only
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.learning_rate = learning_rate
        self.write_attention = write_attention

        if self.pooling_kernel < 0:
            raise ValueError(f"pooling_kernel must be non-negative, found {pooling_kernel}")
        if self.stream_pooling_kernel and self.pooling_kernel == 0:
            raise ValueError(f"stream_pooling_kernel cannot be True when pooling_kernel=0")

        assert encoder in list(StreamingCLAM.model_choices.keys())

        # Define the streaming network and head
        if encoder in ("resnet18", "resnet34", "resnet50"):
            network = StreamingCLAM.model_choices[encoder](weights="IMAGENET1K_V1")
            stream_net, _ = split_resnet(network) #
        head = CLAMConfig(encoder=encoder, branch=branch, n_classes=n_classes).configure_clam()
        # At the end of the ResNet model, reduce the spatial dimensions with additional pooling layers
        self._get_streaming_options(**kwargs)

        self.ds_blocks = None
        if self.pooling_kernel > 0:
            if self.stream_pooling_kernel:
                stream_net = self.add_pooling_layers(stream_net)
                super().__init__(
                    stream_net,
                    head,
                    tile_size,
                    loss_fn,
                    train_streaming_layers=train_streaming_layers,
                    **self.streaming_options,
                )
            else:
                ds_blocks, head = self.add_pooling_layers(head)
                super().__init__( # streaming initialized here
                    stream_net,
                    head,
                    tile_size,
                    loss_fn,
                    train_streaming_layers=train_streaming_layers,
                    **self.streaming_options,
                )
                self.ds_blocks = ds_blocks
        else:
            super().__init__(
                stream_net,
                head,
                tile_size,
                loss_fn,
                train_streaming_layers=train_streaming_layers,
                **self.streaming_options,
            )

        self.train_acc = Accuracy(task="binary" if n_classes <=2 else "multiclass", num_classes=n_classes)
        self.train_binacc = Accuracy(task="binary", num_classes=2)
        self.train_auc = AUROC(task="binary", num_classes=n_classes)

        self.val_acc = Accuracy(task="binary" if n_classes <=2 else "multiclass", num_classes=n_classes)
        self.val_binacc = Accuracy(task="binary", num_classes=2)
        self.val_auc = AUROC(task="binary", num_classes=n_classes)

        self.test_acc = Accuracy(task="binary" if n_classes <=2 else "multiclass", num_classes=n_classes)
        self.test_binacc = Accuracy(task="binary", num_classes=2)
        self.test_auc = AUROC(task="binary", num_classes=n_classes)

        self.test_outputs = []

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

    def _configure_pooling_layer(self):
        if self.pooling_layer == "maxpool":
            pooling_layer = torch.nn.MaxPool2d
        elif self.pooling_layer == "avgpool":
            pooling_layer = torch.nn.AvgPool2d
        else:
            raise TypeError(f'pooling_layer must be one of "maxpool" or "avgpool" but found {self.pooling_layer}')

        return pooling_layer

    def add_pooling_layers(self, network):
        pooling_layer = self._configure_pooling_layer()
        ds_blocks = torch.nn.Sequential(pooling_layer((self.pooling_kernel, self.pooling_kernel), ceil_mode=True))

        if self.stream_pooling_kernel:
            return torch.nn.Sequential(network, ds_blocks)
        else:
            return ds_blocks, network

    def forward_head(
        self,
        fmap,
        mask=None,
        instance_eval=False,
        label=None,
        return_features=False,
        attention_only=False,
    ):
        batch_size, num_features, h, w = fmap.shape

        if self.ds_blocks is not None:
            fmap = self.ds_blocks(fmap)

        # Mask background, can heavily reduce inputs to clam network
        if mask is not None:
            fmap = torch.masked_select(fmap, mask)
            del mask

        # Put everything back together into an array [channels, #unmasked_pixels]
        # Change dimensions from [batch_size, C, H, W] to [batch_size, C, H * W]
        fmap = torch.reshape(fmap, (num_features, -1)).transpose(0, 1)

        if self.attention_only:
            return self.head(
                fmap,
                label=None,
                instance_eval=False,
                attention_only=self.attention_only,
            )

        logits, Y_prob, Y_hat, A_raw, instance_dict = self.head(
            fmap,
            label=label,
            instance_eval=instance_eval,
            return_features=return_features,
            attention_only=attention_only,
        )

        return logits, Y_prob, Y_hat, A_raw, instance_dict

    def forward(self, image, is_embedding, mask=None):
        fmap = self.get_embedding(image,is_embedding,mask.device)
        self.str_output = fmap
        out = self.forward_head(
            fmap,
            mask=mask,
            return_features=self.return_features,
            attention_only=self.attention_only,
        )
        return out

    def get_embedding(self,image,is_embedding,device="cpu"):
        # compute embedding if not computed
        if not is_embedding:
            self.image = image 
            self.embedding_computed = True 
            image = self.forward_streaming(image)
        else:
            self.embedding_computed = False  
            self.image = None
            image = image.to(device)
        return image

    def training_step(self, batch, batch_idx: int, *args, **kwargs):
        image = batch["image"]
        image = image.to("cpu")
        self.mask = batch["mask"] if "mask" in batch.keys() else None
        self.width = batch["width"]
        label = batch["label"]
        is_embedding = torch.all(batch["is_embedding"])
        
        self.str_output = self.get_embedding(image, is_embedding,self.mask.device)
        self.str_output.requires_grad = self.training

        logits, Y_prob, Y_hat, A_raw, instance_dict = self.forward_head(
            self.str_output,
            mask=self.mask,
            instance_eval=self.instance_eval,
            label=label if self.instance_eval else None,
            return_features=self.return_features,
            attention_only=self.attention_only,
        )
        """ ADD METRICS """
        loss = self.loss_fn(logits, label)
        probs = torch.nn.functional.softmax(logits, dim=1)
        # handle non-binary probabilities
        binary_probs = probs[:,1:].sum(dim=1)
        self.train_acc.update(torch.argmax(logits, dim=1).detach(), label.detach())
        self.train_auc.update(binary_probs.detach(), label.detach())
        # For binary accuracy
        binpred = (torch.argmax(logits, dim=1) >= 1) * 1  # Consider logits >= 1 as 1, otherwise 0
        binlabel = (label >= 1) * 1 # Consider label >= 1 as 1, otherwise 0
        self.train_binacc.update(binpred.detach(), binlabel.detach())

        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/bacc", self.train_binacc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/auc", self.train_auc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, label = self._shared_eval_step(batch, batch_idx)
        del batch

        probs = torch.nn.functional.softmax(logits, dim=1)
        # handle non-binary probabilities
        binary_probs = probs[:,1:].sum(dim=1)
        self.val_acc.update(torch.argmax(logits, dim=1).detach(), label)
        self.val_auc.update(binary_probs, label)
        # For binary accuracy
        binpred = (torch.argmax(logits, dim=1) >= 1) * 1  # Consider logits >= 1 as 1, otherwise 0
        binlabel = (label >= 1) * 1 # Consider label >= 1 as 1, otherwise 0
        self.val_binacc.update(binpred.detach(), binlabel.detach())
        # Should update and clear automatically, as per
        # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        # https: // lightning.ai / docs / pytorch / stable / extensions / logging.html

        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/bacc", self.val_binacc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/auc", self.val_auc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, label = self._shared_eval_step(batch, batch_idx)

        probs = torch.nn.functional.softmax(logits, dim=1)
        # handle non-binary probabilities
        binary_probs = probs[:,1:].sum(dim=1)
        self.test_acc(torch.argmax(logits, dim=1), label)
        self.test_auc(binary_probs, label)
        # For binary accuracy
        binpred = (torch.argmax(logits, dim=1) >= 1) * 1  # Consider logits >= 1 as 1, otherwise 0
        binlabel = (label >= 1) * 1 # Consider label >= 1 as 1, otherwise 0
        self.test_binacc.update(binpred.detach(), binlabel.detach())

        metrics = {
            "test/acc": self.test_acc,
            "test/bacc": self.test_binacc,
            "test/auc": self.test_auc,
            "test/loss": loss,
        }
        self.log_dict(metrics, prog_bar=True)

        self.test_outputs.append(
            {
                "slide_name": batch["image_name"],
                "loss": float(loss.detach().cpu().numpy()),
                "probs": binary_probs.detach().cpu().numpy().squeeze(),
                "y_hat": float(torch.argmax(logits, dim=1).detach().cpu().numpy()),
                "label": int(label.cpu().numpy()),
            }
        )
        return loss.detach().cpu()

    def _shared_eval_step(self, batch, batch_idx):
        image = batch["image"]
        image = image.to("cpu")
        self.mask = batch["mask"] if "mask" in batch.keys() else None
        self.width = batch["width"]
        label = batch["label"]
        is_embedding = torch.all(batch["is_embedding"])

        logits, Y_prob, Y_hat, A_raw, instance_dict = self.forward(image, is_embedding, mask=self.mask)
        loss = self.loss_fn(logits, label)
        return loss, logits.detach(), label.detach()

    def _get_streaming_options(self, **kwargs):
        """Set streaming defaults, but overwrite them with values of kwargs if present."""

        # We need to add torch.nn.Batchnorm to the keep modules, because of some in-place ops error if we don't
        # https://discuss.pytorch.org/t/register-full-backward-hook-for-residual-connection/146850
        streaming_options = {
            "verbose": False,
            "copy_to_gpu": False,
            "statistics_on_cpu": True,
            "normalize_on_gpu": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "add_keep_modules": [torch.nn.BatchNorm2d],
        }
        self.streaming_options = {**streaming_options, **kwargs}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.write_attention:
            image = batch["image"]
            image = image.to("cpu")
            self.mask = batch["mask"] if "mask" in batch.keys() else None
            self.width = batch["width"]
            is_embedding = torch.all(batch["is_embedding"])
            logits, Y_prob, Y_hat, A_raw, instance_dict = self.forward(image, is_embedding, mask=self.mask)

            # Add to batch for write_on_batch_end
            batch.update({"A_raw": A_raw})

            # Discard attention map,

            return logits.detach().cpu(), Y_prob.detach().cpu(), Y_hat.detach().cpu()
        else:
            # Just perform a predict with a normal dataloader
            return self.test_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.params, lr=self.learning_rate, weight_decay=1e-5)

        def lr_lambda(epoch):
            if epoch < self.train_streaming_layers:
                return 1
            else:
                # halve the learning rate when switching to training all layers
                return 0.1

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "name": "lr_scheduler",
        }

        return [optimizer], [lr_scheduler]

    def backward(self, loss):
        loss.backward()
        # del loss
        # Don't call this>? https://pytorch-lightning.readthedocs.io/en/1.5.10/guides/speed.html#things-to-avoid
        torch.cuda.empty_cache()
        if self.train_streaming_layers:
            self.backward_streaming(self.image, self.str_output.grad)


if __name__ == "__main__":
    model = StreamingCLAM(
        "resnet18",
        tile_size=1600,
        loss_fn=torch.nn.functional.cross_entropy,
        branch="sb",
        n_classes=4,
        pooling_kernel=8,
    )
