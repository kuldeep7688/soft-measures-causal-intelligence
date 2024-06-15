import torch
import torchmetrics
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from config.config_loader import load_config, CONFIG

class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate, t_total, warmup_steps, num_classes):
        super().__init__()

        # Load config and retrieve weight_decay value
        config = load_config(CONFIG)
        self.weight_decay = float(config['training']['weight_decay'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.t_total = t_total
        self.warmup_steps = warmup_steps
        
        self.save_hyperparameters(ignore=['model'])
        
        # Initialize metrics
        self.train_f1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=num_classes) 
        self.val_f1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=num_classes) 
        self.test_f1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=num_classes)
        self.val_f1_micro = torchmetrics.F1Score(task="multiclass", average='micro', num_classes=num_classes) 
        self.test_f1_micro = torchmetrics.F1Score(task="multiclass", average='micro', num_classes=num_classes)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass of the model."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self(**batch)
        
        # Log training loss
        self.log("train_loss", outputs["loss"], prog_bar=True)
        
        # Compute and log training F1 score
        with torch.no_grad():
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, -1)
            self.train_f1(predicted_labels, batch['labels'])
            self.log("train_f1", self.train_f1, on_epoch=True, on_step=True)
            
        return outputs['loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self(**batch)
        
        # Log validation loss
        self.log("val_loss", outputs["loss"], prog_bar=True)
        
        # Compute and log validation F1 score
        logits = outputs['logits']
        predicted_labels = torch.argmax(logits, -1)
        self.val_f1(predicted_labels, batch['labels'])
        self.val_f1_micro(predicted_labels, batch['labels'])
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val_f1_micro", self.val_f1_micro, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs = self(**batch)
        
        logits = outputs['logits']
        predicted_labels = torch.argmax(logits, -1)
        
        # Compute and log test F1 score
        self.test_f1(predicted_labels, batch['labels'])
        self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)
        
        # Compute and log test F1 micro score
        self.test_f1_micro(predicted_labels, batch['labels'])
        self.log("test_f1_micro", self.test_f1_micro, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.t_total
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]
        