import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import torch.multiprocessing as mp
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from config_schema import ModelConfig, DataConfig, TrainerConfig
from src.Models.aacnn import AACNN
from src.Data.data_utils import SpectralDataModule, create_experiment_split


from src.Engine.trainer_engine import run_training
import src.Engine.inference_engine 
from src.Engine.inference_engine import run_inference
from src.Engine.save_inference import save_inference_outputs_zarr, open_pred_store

cs = ConfigStore.instance()
cs.store(group="model", name="aacnn_config", node=ModelConfig)
cs.store(group="data", name="data_config", node=DataConfig)
cs.store(group="trainer", name="trainer_config", node=TrainerConfig)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    model_cfg = hydra.utils.instantiate(cfg.model)
    data_cfg = hydra.utils.instantiate(cfg.data)
    trainer_cfg = hydra.utils.instantiate(cfg.trainer) 

    train_test_split = create_experiment_split(data_cfg.zarr_path, split_ratio=0.85)
    
    
    if cfg.mode in ["train", "all"]:
        model = AACNN(model_cfg)
        datamodule = SpectralDataModule(train_test_split['train'], data_cfg)
        datamodule.setup()
        datamodule.label_encoding
        
        best_path = run_training(cfg, model, datamodule)
        cfg.ckpt_path = best_path  
        
    if cfg.mode in ["infer", "all"]:
        if not cfg.ckpt_path:
            raise ValueError("You must provide a 'ckpt_path' to run inference!")
   
        pred_store  = open_pred_store(cfg.pred_store_path)   
        
        test_images = train_test_split['test']
        print(f"Starting batch inference on {len(test_images)} images...")
        
        for img_data in tqdm(test_images, desc="Inference Progress", unit="image"):
            
            try:
                prob_map = run_inference(cfg, image_name=img_data['name'], ckpt_path=cfg.ckpt_path)

                save_inference_outputs_zarr(
                    prob_map       = prob_map,
                    image_name     = img_name,
                    store          = pred_store,
                    N              = cfg.model.N,
                    seed           = cfg.seed,
                    background_idx = cfg.background_idx,
                    top_k_save     = cfg.top_k_save,
                    true_idx       = datamodule.label_encoding[img_data['label']],
                    hparams        = {
                        "lr":         cfg.model.lr,
                        "batch_size": cfg.data.batch_size,
                    },
                )

            except Exception as e:
                print(f"[WARN] {img_data['name']} failed: {e}")
                failed.append(img_data['name'])

        pred_store.store.close()   

        
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()