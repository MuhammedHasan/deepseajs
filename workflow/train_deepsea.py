import h5py
import torch
import pytorch_lightning as pl
from .deepsea import DeepSea, DeepSeaModule

torch.set_float32_matmul_precision('medium')

threads = snakemake.threads
devices = snakemake.params['devices']

output_size = h5py.File(snakemake.input['train'])['traindata'].shape[0]
model = DeepSea(output_size=output_size)

datamodule = DeepSeaModule(
    snakemake.input['train'],
    snakemake.input['val'],
    snakemake.input['test'],
    num_workers=threads,
    batch_size=snakemake.params['batch_size']
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=snakemake.output['checkpoint'],
    save_top_k=1,
    monitor='val_loss',
    mode='min'
)

trainer = pl.Trainer(
    devices=devices,
    max_epochs=snakemake.params['epochs'],
    callbacks=[
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=snakemake.params['early_stopping'],
            mode='min'
        ),
        checkpoint_callback
    ],
    logger=pl.loggers.WandbLogger(
        project='deepsea',
    ),
    precision="bf16-mixed",
)
trainer.fit(model=model, datamodule=datamodule)

best_model = DeepSea.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    output_size=output_size, dropout=0)

torch.save(best_model.state_dict(), snakemake.output['model'])
