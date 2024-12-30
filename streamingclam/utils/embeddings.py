import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import os
from pathlib import Path
import shutil
from tqdm import tqdm
from torch.distributed import barrier, is_initialized

def sync_all_ranks():
    if is_initialized():
        barrier()

class IntermediateEmbeddings(BasePredictionWriter):
    def __init__(
        self,
        embeddings_source: Path,
        use_embeddings: bool = False,
        unfreeze_at_epoch : int = 10,
        embeddings_temp_dir : Path = Path("/home/embeddings"),
        export_to_remote_every : int = 50,
    ):
        super().__init__()
        self.embeddings_source = embeddings_source
        self.use_embeddings = use_embeddings
        self.save_embeddings = self.use_embeddings
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.embeddings_temp_dir = embeddings_temp_dir
        self.export_to_remote_every = export_to_remote_every

        self.new_local_embedding = False # switch to know whether to export embeddings to remote or not
        self.train_batch_size = None
        self.val_batch_size = None 
        self.backup_directly = False # switch to exporting embeddings to remote after every computation

    def write_on_batch_end(self,**kwargs):
        pass

    def write_on_epoch_end(self,**kwargs):
        pass

    def on_val_batch_end(self,**kwargs):
        pass
   
    def on_predict_epoch_end(self,**kwargs):
        pass

    @rank_zero_only
    def move_unique_files(self,src_dir = Path(""), dst_dir = Path(""),verbose=False):
        src_files = set(os.listdir(src_dir))
        dst_files = set(os.listdir(dst_dir))
        new_files = src_files - dst_files
        
        if len(new_files) > 0 :
            for file_path in tqdm(new_files,total=len(new_files),disable=not verbose):
                shutil.copyfile(src_dir / file_path, dst_dir / file_path)

        """alternatively ( not recommended ) : move the full directory """
         # copytree(str(self.embeddings_temp_dir),str(self.embeddings_source),dirs_exist_ok=True)
 
    def write_embedding(self,image_name, embedding,mask,width,batch_idx,backup_directly=False):
        _out = {"embedding": embedding.detach().cpu().squeeze(0),
            "mask": mask,
            "width": width
        }

        cache_path = self.embeddings_temp_dir /  f"{image_name}.pt"
        torch.save(_out, cache_path) 

        """ Decide whether to save on remove on every save operation in save_batch. alternatively, move_unique_files periodically  """
        if backup_directly:
            cache_path = self.embeddings_source /  f"{image_name}.pt"
            torch.save(_out, cache_path) 

        del _out

    # def on_fit_start(self,trainer,pl_module): fails to load the current epoch from the checkpoint
    def on_train_start(self,trainer,pl_module):
        # if unfreeze at epoch 21, then we unfreeze from current_epoch 20. when current_epoch < 20, we use embeddings
        # i.e when current_epoch < unfreeze - 1
        # when encoder is frozen and intermediate embeddings are used 
        if (trainer.current_epoch < (self.unfreeze_at_epoch -1 )) and self.use_embeddings: # unfreeze = 20 , current = 19, 
            # copy existing embeddings from remote
            print("============== start using intermediate embeddings ================" )
            os.makedirs(self.embeddings_temp_dir, exist_ok=True)
            """optionally move embeddings from remove to local at the start of training. NOT RECOMMENDED to do this in the callback as it causes NCCL unsync errors"""
            self.move_unique_files(Path(self.embeddings_source),Path(self.embeddings_temp_dir),verbose=True)
            sync_all_ranks()
            # instruct dataloader to load any precomputed embeddings
            trainer.datamodule.load_embeddings = self.use_embeddings
            trainer.datamodule.embeddings_source = self.embeddings_temp_dir
            trainer.datamodule.reset("fit")


    def on_test_start(self,trainer,pl_module):
        self.on_train_start(trainer,pl_module)

    def on_predict_start(self,trainer,pl_module):
        self.on_train_start(trainer,pl_module)

    def save_batch(self,pl_module,embeddings,batch_idx,image_name,backup_directly=False):
        # save embeddings locally
        if self.save_embeddings and pl_module.embedding_computed:
            for _item in embeddings:
                embedding = _item.unsqueeze(0)
                image_name = image_name
                mask = pl_module.mask
                width = pl_module.width

                self.write_embedding(image_name,embedding,mask,width, batch_idx,backup_directly=backup_directly)
            
            self.new_local_embedding = True

        if not backup_directly:
            if self.new_local_embedding and ((batch_idx+1) % self.export_to_remote_every) == 0:
                if not self.embeddings_temp_dir == self.embeddings_source:
                    self.move_unique_files(self.embeddings_temp_dir,self.embeddings_source)
                    sync_all_ranks()
                self.new_local_embedding = False

    def on_train_batch_end(self,trainer, pl_module, output, batch, batch_idx):
        image_name = batch["image_name"]
        self.save_batch(pl_module,pl_module.str_output,batch_idx,image_name,backup_directly=self.backup_directly)
        del pl_module.str_output, pl_module.image

    def on_validation_batch_end(self,trainer, pl_module, output, batch, batch_idx):
        self.on_train_batch_end(trainer,pl_module,output,batch,batch_idx)
    
    def on_test_batch_end(self,trainer, pl_module, output, batch, batch_idx):
        self.on_train_batch_end(trainer,pl_module,output,batch,batch_idx)

    def on_predict_batch_end(self,trainer, pl_module, output, batch, batch_idx):
        self.on_train_batch_end(trainer,pl_module,output,batch,batch_idx)

    def on_train_end(self,trainer, pl_module):
        if self.save_embeddings:
            if not self.embeddings_temp_dir == self.embeddings_source:
                self.move_unique_files(self.embeddings_temp_dir,self.embeddings_source,verbose=True)
                sync_all_ranks()
            
            self.new_local_embedding = False

    def on_test_end(self,trainer, pl_module):
        self.on_train_end(trainer,pl_module)
                
    def on_validation_end(self,trainer, pl_module):
        self.on_train_end(trainer,pl_module)

        """ stop using intermediate embeddings when unfreezing the encoder """
        if trainer.current_epoch == (self.unfreeze_at_epoch-1):
            print("============== stop using intermediate embeddings ================" )
            self.use_embeddings = False
            self.save_embeddings = False
            self.new_local_embedding = False

            trainer.datamodule.load_embeddings = False
            trainer.datamodule.verbose = False
            trainer.datamodule.reset("fit")  


 

