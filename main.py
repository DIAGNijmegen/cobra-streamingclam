import os

os.environ["WANDB_DIR"] = "/home/ivanslootweg/data/SBCC"

import pyvips

pyvips.cache_set_max(20)
pyvips.cache_set_max_mem(1024 * 1024)

import torch
import warnings

from pathlib import Path
from tqdm import tqdm 

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger

from streamingclam.options import TrainConfig
from streamingclam.utils.memory_format import MemoryFormat
from streamingclam.utils.printing import PrintingCallback
from streamingclam.utils.finetune import FeatureExtractorFreezeUnfreeze
from streamingclam.utils.embeddings import IntermediateEmbeddings
from streamingclam.data.splits import StreamingCLAMDataModule
from streamingclam.data.dataset import augmentations
from streamingclam.models.sclam import StreamingCLAM
from streamingclam.utils.writers import AttentionWriter, TestPredictionWriter

torch.set_float32_matmul_precision("medium")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def configure_callbacks(options):
    callbacks = []
    if options.mode == "fit":
        checkpoint_callback = ModelCheckpoint(
            dirpath=options.default_save_dir + f"/{options.experiment_name}/fold_{options.fold}/ckp",
            monitor="val/loss",
            filename="streamingclam-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
            save_top_k=3,
            save_last=True,
            mode="min",
            verbose=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        embeddings_cb = IntermediateEmbeddings(
            embeddings_source = Path(options.embeddings_source),
            load_embeddings = options.load_embeddings,
            save_embeddings = options.save_embeddings,
            unfreeze_at_epoch = options.unfreeze_streaming_layers_at_epoch,
            embeddings_temp_dir = Path(options.embeddings_temp_dir),
            export_to_remote_every = 200,
        )

        finetune_cb = FeatureExtractorFreezeUnfreeze(
            options.unfreeze_streaming_layers_at_epoch,
            tile_size_finetune=options.tile_size_finetune,
            lambda_func=lambda epoch: 5,
            verbose = True
        )
        memory_format_cb = MemoryFormat()
        print_cb = PrintingCallback(options)
        if options.save_embeddings or options.load_embeddings:
            callbacks = [checkpoint_callback, lr_monitor, finetune_cb,embeddings_cb,memory_format_cb, print_cb]
        else:
            callbacks = [checkpoint_callback, lr_monitor, finetune_cb,embeddings_cb,memory_format_cb, print_cb]
    elif options.mode=="attention":
        writer_cb = AttentionWriter(Path(options.default_save_dir) / Path(f"{options.experiment_name}/attentions"),
                                    read_level=options.read_level,
                                    write_level=options.write_level,
                                    write_interval="batch" if options.mode=="attention" else "epoch")
        callbacks = [writer_cb]
    elif options.mode=="test":
        test_writer = TestPredictionWriter(Path(options.default_save_dir + f"/{options.experiment_name}/fold_{str(options.fold)}"))
        callbacks = [test_writer]
    return callbacks


def configure_checkpoints():
    try:
        # Check for last checkpoint
        if options.resume_epoch :
            checkpoint = list(Path(options.default_save_dir + f"/{options.experiment_name}/fold_{str(options.fold)}/ckp").glob(f"*{options.resume_epoch}-val_loss*.ckpt"))
            checkpoint_path = str(checkpoint[0])          
        elif options.resume:
            checkpoint = list(Path(options.default_save_dir + f"/{options.experiment_name}/fold_{str(options.fold)}/ckp").glob("*last.ckpt"))
            checkpoint_path = str(checkpoint[0])
        else:
            checkpoint_path = None
    except IndexError:
        if options.resume:
            warnings.warn("Resume option enabled, but no checkpoint files found. Training will start from scratch.")
        checkpoint_path = None

    return checkpoint_path


def configure_trainer(options, wandb_logger=None):
    callbacks = configure_callbacks(options)
    trainer = pl.Trainer(
        default_root_dir=options.default_save_dir,
        accelerator="gpu" if options.num_gpus >0 else "cpu",
        max_epochs=options.num_epochs,
        devices=max(1,options.num_gpus),
        accumulate_grad_batches=options.grad_batches,
        precision=options.precision,
        callbacks=callbacks,
        strategy=options.strategy,
        benchmark=False,
        reload_dataloaders_every_n_epochs=options.unfreeze_streaming_layers_at_epoch,
        logger=wandb_logger if wandb_logger else None,  
    )
    return trainer


def get_model_statistics(model):
    """Prints model statistics for reference purposes

    Prints network output strides, and tile delta for streaming

    Parameters
    ----------
    model : pytorch lightning model object


    """

    tile_stride = model.configure_tile_stride()
    network_output_stride = model.stream_network.output_stride[1]
    return tile_stride, network_output_stride


def get_streaming_options(options):
    fields = [
        "statistics_on_cpu",
        "normalize_on_gpu",
        "copy_to_gpu",
        "verbose",
    ]
    opt_dict = options.to_dict()
    return {key: opt_dict[key] for key in fields}


def configure_streamingclam(options, streaming_options):
    sclam_opts = {
        "encoder": options.encoder,
        "tile_size": options.tile_size,
        "loss_fn": options.loss_fn,
        "branch": options.branch,
        "n_classes": options.num_classes,
        "pooling_layer": options.pooling_layer,
        "pooling_kernel": options.pooling_kernel,
        "stream_pooling_kernel": options.stream_pooling_kernel,
        "train_streaming_layers": options.train_streaming_layers,
        "instance_eval": options.instance_eval,
        "return_features": options.return_features,
        "attention_only": options.attention_only,
        "unfreeze_at_epoch": options.unfreeze_streaming_layers_at_epoch,
        "learning_rate": options.learning_rate,
        "write_attention": True,
        "additive": options.additive_mil
    }

    if options.mode == "fit":
        model = StreamingCLAM(
            **sclam_opts,
            **streaming_options,
        )
    else:
        model = StreamingCLAM.load_from_checkpoint(
            options.ckp_path,
            **sclam_opts,
            **streaming_options,
        )
    return model


def configure_datamodule(options):
    return StreamingCLAMDataModule(
        image_dir=options.image_path,
        level=options.read_level,
        tile_size=options.tile_size,
        tile_stride=options.tile_stride,
        network_output_stride=options.network_output_stride,
        train_csv_path=options.train_csv,
        val_csv_path=options.val_csv,
        test_csv_path=options.test_csv,
        attention_csv_path=options.attention_csv,
        tissue_mask_dir=options.mask_path if len(options.mask_path) > 0 else None,
        mask_suffix=options.mask_suffix,
        image_size=options.image_size,
        variable_input_shapes=options.variable_input_shapes,
        copy_to_gpu=options.copy_to_gpu,
        num_workers=options.num_workers,
        transform=augmentations if (not options.use_augmentations and options.mode == "fit") else None,
        output_dir=Path(options.default_save_dir) / Path(f"/{options.experiment_name}/attentions"),
        embeddings_source=Path(options.embeddings_temp_dir),
        load_embeddings=options.load_embeddings,
        streaming_embeddings = options.streaming_embeddings,
        embedding_extension = options.embedding_extension
    )


def get_options():
    # Read json config from file
    options = TrainConfig()
    parser = options.configure_parser_with_options()
    args = parser.parse_args()
    print(args)
    options.parser_to_options(vars(args))

    return options


if __name__ == "__main__":
    pl.seed_everything(1)

    options = get_options()
    streaming_options = get_streaming_options(options)

    model = configure_streamingclam(options, streaming_options)
    tile_stride, network_output_stride = get_model_statistics(model)
    options.tile_stride = tile_stride

    if options.stream_pooling_kernel:
        options.network_output_stride = network_output_stride
    else:
        options.network_output_stride = max(network_output_stride * options.pooling_kernel, network_output_stride)
    dm = configure_datamodule(options)
    dm.setup(stage=options.mode)


    if options.mode == "fit":
        wandb_logger = WandbLogger(
            name=options.experiment_name,
            project=options.wandb_project_name,
            save_dir=options.default_save_dir,
        )
        logger = TensorBoardLogger(options.default_save_dir, version=f"{options.experiment_name}_{options.fold}", name="lightning_logs")
        loggers = [wandb_logger,logger]
        trainer = configure_trainer(options,loggers)
        checkpoint_path = configure_checkpoints()
        # model.head = torch.compile(model.head)
        # model.stream_network.stream_module = torch.compile(model.stream_network.stream_module)
        # print(model.stream_network)
        trainer.fit(
            model=model,
            datamodule=dm,
            ckpt_path=checkpoint_path
        )
            
    elif options.mode=="attention" or options.mode=="test":
        trainer = configure_trainer(options)
        if options.mode=="attention":
            trainer.predict(model=model, datamodule=dm,)
        elif options.mode=="test":
            trainer.test(model=model, datamodule=dm,)



    else:
        raise ValueError("mode must be one of fit, test, attention, or predict, found {}".format(options.mode))

# DO:
