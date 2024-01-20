import argparse
import os

import nemo.collections.asr as nemo_asr
import pytorch_lightning as ptl
from nemo.utils import exp_manager, logging
from omegaconf import OmegaConf, open_dict


def train_model(
        train_manifest: str, 
        val_manifest: str, 
        accelerator: str = "cpu", 
        batch_size: int = 1, 
        num_epochs: int = 1, 
        model_save_path: str = None,
    ) -> None:

    # Loading a STT Quartznet 15x5 model
    model = nemo_asr.models.ASRModel.from_pretrained("stt_en_quartznet15x5")
    
    # New vocabulary for a model
    new_vocabulary = [
        " ",
        "а",
        "б",
        "в",
        "г",
        "д",
        "е",
        "ж",
        "з",
        "и",
        "й",
        "к",
        "л",
        "м",
        "н",
        "о",
        "п",
        "р",
        "с",
        "т",
        "у",
        "ф",
        "х",
        "ц",
        "ч",
        "ш",
        "щ",
        "ъ",
        "ы",
        "ь",
        "э",
        "ю",
        "я",
        "і",
        "ғ",
        "қ",
        "ң",
        "ү",
        "ұ",
        "һ",
        "ә",
        "ө",
    ]
    
    # Configurations
    with open_dict(model.cfg):
        # Setting up the labels and sample rate
        model.cfg.labels = new_vocabulary
        model.cfg.sample_rate = 16000

        # Train dataset
        model.cfg.train_ds.manifest_filepath = train_manifest
        model.cfg.train_ds.labels = new_vocabulary
        model.cfg.train_ds.normalize_transcripts = False
        model.cfg.train_ds.batch_size = batch_size
        model.cfg.train_ds.num_workers = 10
        model.cfg.train_ds.pin_memory = True
        model.cfg.train_ds.trim_silence = True

        # Validation dataset
        model.cfg.validation_ds.manifest_filepath = val_manifest
        model.cfg.validation_ds.labels = new_vocabulary
        model.cfg.validation_ds.normalize_transcripts = False
        model.cfg.validation_ds.batch_size = batch_size
        model.cfg.validation_ds.num_workers = 10
        model.cfg.validation_ds.pin_memory = True
        model.cfg.validation_ds.trim_silence = True

        # Setting up an optimizer and scheduler
        model.cfg.optim.lr = 0.001
        model.cfg.optim.betas = [0.8, 0.5]
        model.cfg.optim.weight_decay = 0.001
        model.cfg.optim.sched.warmup_steps = 500
        model.cfg.optim.sched.min_lr = 1e-6

    model.change_vocabulary(new_vocabulary=new_vocabulary)
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    # Unfreezing encoders to update the parameters
    model.encoder.unfreeze()
    logging.info("Model encoder has been un-frozen")

    # Setting up data augmentation
    model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)

    # Setting up the metrics
    model._wer.use_cer = True
    model._wer.log_prediction = True

    # Trainer
    trainer = ptl.Trainer(
        accelerator=accelerator,
        max_epochs=num_epochs,
        accumulate_grad_batches=1,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=100,
        check_val_every_n_epoch=1,
        precision=16,
    )

    # Setting up model with the trainer
    model.set_trainer(trainer)

    # Experiment tracking
    LANGUAGE = "kz"
    config = exp_manager.ExpManagerConfig(
        exp_dir=f"experiments/lang-{LANGUAGE}/",
        name=f"ASR-Model-Language-{LANGUAGE}",
        checkpoint_callback_params=exp_manager.CallbackParams(monitor="val_wer", mode="min", always_save_nemo=True, save_best_model=True,),
    )
    config = OmegaConf.structured(config)
    exp_manager.exp_manager(trainer, config)

    # Final Configuration
    print("-----------------------------------------------------------")
    print("Updated STT Model Configuration:")
    print(OmegaConf.to_yaml(model.cfg))
    print("-----------------------------------------------------------")

    # Fitting the model
    trainer.fit(model)

    # Saving the model
    if model_save_path:
        model.save_to(f"{model_save_path}")
        print(f"Model saved at path : {os.getcwd() + os.path.sep + model_save_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", help="Path for train manifest JSON file.")
    parser.add_argument("--val_manifest", help="Path for validation manifest JSON file.")
    parser.add_argument("--accelerator", default="cpu", help="What accelerator type to use (cpu, gpu, tpu, etc.).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of the dataset to train.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for.")
    parser.add_argument("--model_save_path", default=None, help="Path for saving a trained model.")
    args = parser.parse_args()

    train_model(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        accelerator=args.accelerator,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        model_save_path=args.model_save_path,
    )
