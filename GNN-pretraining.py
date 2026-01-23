import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import json
import gc
from datetime import datetime
from GNNmodel import create_model
from GNNdataset import ChEMBLDataset, create_dataloaders
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1        
        if self.counter >= self.patience:
            self.early_stop = True        
        return self.early_stop
class ChEMBLPretrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print("Initializing model...")
        self.model = create_model(config['model']).to(self.device)
        self.model.apply(self._init_batchnorm)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params:,}")
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        self.scheduler_cosine = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['scheduler_T0'],
            T_mult=config['scheduler_Tmult'],
            eta_min=config['min_lr']
        )        
        self.scheduler_plateau = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=config['min_lr']
        )
        self.scaler = torch.amp.GradScaler('cuda') if config['use_amp'] and torch.cuda.is_available() else None
        self.early_stopping = EarlyStopping(
            patience=config['patience'],
            min_delta=0.001,
            mode='min'
        )
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_val_gap = []
        self.lr_reduction_count = 0
        self.max_lr_reductions = config.get('max_lr_reductions', 5)
        self.current_weight_decay = config['weight_decay']
        self.initial_weight_decay = config['weight_decay']
        self.lr_history = []
        self.wd_history = []
        self.issue_history = {}    
    def _init_batchnorm(self, m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = 0.1
            m.eps = 1e-5
            m.track_running_stats = True
            if m.running_mean is not None:
                m.running_mean.zero_()
            if m.running_var is not None:
                m.running_var.fill_(1)        
    def adjust_weight_decay(self, train_loss, val_loss):
        gap = train_loss - val_loss
        if gap < -0.5 and self.current_weight_decay < 0.1:
            self.current_weight_decay *= 2.0
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = self.current_weight_decay
            print(f"  ⚠️  Overfitting detected! Increased weight decay to {self.current_weight_decay:.2e}")
            return True
        elif train_loss > 1.0 and abs(gap) < 0.1 and self.current_weight_decay > 1e-6:
            self.current_weight_decay *= 0.5
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = self.current_weight_decay
            print(f"  ℹ️  Underfitting detected! Decreased weight decay to {self.current_weight_decay:.2e}")
            return True        
        return False    
    def adjust_dropout(self, train_loss, val_loss):
        gap = train_loss - val_loss
        if gap < -0.5:
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    if module.p < 0.5:
                        module.p = min(0.5, module.p * 1.2)
            print(f"  📊 Increased dropout rate")
            return True
        elif train_loss > 1.0 and abs(gap) < 0.1:
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    if module.p > 0.05:
                        module.p = max(0.05, module.p * 0.8)
            print(f"  📊 Decreased dropout rate")
            return True        
        return False    
    def detect_training_issues(self, train_loss, val_loss, epoch):
        issues = []
        gap = train_loss - val_loss
        if gap < -1.0:
            issues.append("SEVERE_OVERFITTING")
        elif gap < -0.5:
            issues.append("OVERFITTING")
        if train_loss > 2.0 and val_loss > 2.0:
            issues.append("SEVERE_UNDERFITTING")
        elif train_loss > 1.0 and abs(gap) < 0.2:
            issues.append("UNDERFITTING")
        if epoch > 5:
            recent_train = self.train_losses[-5:]
            if all(recent_train[i] < recent_train[i+1] for i in range(4)):
                issues.append("TRAINING_DIVERGENCE")
        if epoch > 10:
            recent_val = self.val_losses[-10:]
            if max(recent_val) - min(recent_val) < 0.05:
                issues.append("TRAINING_PLATEAU")        
        return issues    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        skipped_batches = 0        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            if batch is None:
                skipped_batches += 1
                continue            
            batch = batch.to(self.device)
            if torch.isnan(batch.y).any() or torch.isinf(batch.y).any():
                skipped_batches += 1
                continue
            pred = self.model(batch).squeeze()
            pred = torch.clamp(pred, min=-25.0, max=5.0)
            loss = self.criterion(pred, batch.y.squeeze())
            if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(pred).any() or torch.isinf(pred).any():
                skipped_batches += 1
                continue
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                skipped_batches += 1
                continue
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            if num_batches > 0:
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
            if num_batches % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()        
        if skipped_batches > 0:
            print(f"  ⚠️  Skipped {skipped_batches}/{skipped_batches + num_batches} training batches")        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_targets = []
        skipped_batches = 0
        debug_first_batch = (epoch == 1)        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
            for batch_idx, batch in enumerate(pbar):
                if batch is None:
                    skipped_batches += 1
                    continue                
                try:
                    batch = batch.to(self.device)
                    if torch.isnan(batch.y).any() or torch.isinf(batch.y).any():
                        if debug_first_batch and batch_idx == 0:
                            print(f"\n  DEBUG: Batch {batch_idx} has NaN/inf targets")
                        skipped_batches += 1
                        continue
                    pred = self.model(batch).squeeze()
                    pred = torch.clamp(pred, min=-25.0, max=5.0)
                    loss = self.criterion(pred, batch.y.squeeze())
                    if debug_first_batch and batch_idx == 0:
                        print(f"\n  DEBUG Batch 0:")
                        print(f"    Targets: min={batch.y.min():.2f}, max={batch.y.max():.2f}, mean={batch.y.mean():.2f}")
                        print(f"    Preds: min={pred.min():.2f}, max={pred.max():.2f}, mean={pred.mean():.2f}")
                        print(f"    Loss: {loss.item():.4f}")
                        print(f"    Has NaN pred: {torch.isnan(pred).any()}")
                        print(f"    Has inf pred: {torch.isinf(pred).any()}")
                    if torch.isnan(loss) or torch.isinf(loss):
                        if debug_first_batch and batch_idx < 5:
                            print(f"  DEBUG: Batch {batch_idx} has NaN/inf loss")
                        skipped_batches += 1
                        continue
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        if debug_first_batch and batch_idx < 5:
                            print(f"  DEBUG: Batch {batch_idx} has NaN/inf predictions")
                        skipped_batches += 1
                        continue                    
                    total_loss += loss.item()
                    num_batches += 1                    
                    all_preds.extend(pred.cpu().numpy().tolist())
                    all_targets.extend(batch.y.squeeze().cpu().numpy().tolist())                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    skipped_batches += 1
                    continue        
        if num_batches == 0:
            print(f"  ⚠️  WARNING: All {skipped_batches} validation batches were skipped!")
            print(f"  This indicates data quality issues or model instability.")
            fallback_loss = self.val_losses[-1] if self.val_losses else 10.0
            return fallback_loss, fallback_loss, fallback_loss, 0.0        
        if skipped_batches > 0:
            print(f"  ⚠️  Skipped {skipped_batches}/{skipped_batches + num_batches} validation batches")        
        avg_loss = total_loss / num_batches
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)        
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0        
        return avg_loss, mae, rmse, r2    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_cosine_state_dict': self.scheduler_cosine.state_dict(),
            'scheduler_plateau_state_dict': self.scheduler_plateau.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()    
    def train(self, train_loader, val_loader):
        print("\n" + "="*80)
        print("Starting ChEMBL Pretraining (ΔG Prediction)")
        print("Target: ΔG in kcal/mol (range: -20 to 0)")
        print("="*80)        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            print("-" * 80)
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            val_loss, val_mae, val_rmse, val_r2 = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)
            gap = train_loss - val_loss
            self.train_val_gap.append(gap)
            issues = self.detect_training_issues(train_loss, val_loss, epoch)
            adjustments_made = False
            old_lr = self.optimizer.param_groups[0]['lr']            
            if "OVERFITTING" in issues or "SEVERE_OVERFITTING" in issues:
                print(f"  🔴 Overfitting detected (gap: {gap:.4f})")
                # Increase regularization
                if self.adjust_weight_decay(train_loss, val_loss):
                    adjustments_made = True
                if self.adjust_dropout(train_loss, val_loss):
                    adjustments_made = True            
            if "UNDERFITTING" in issues or "SEVERE_UNDERFITTING" in issues:
                print(f"  🟡 Underfitting detected")
                if self.adjust_weight_decay(train_loss, val_loss):
                    adjustments_made = True
                if self.adjust_dropout(train_loss, val_loss):
                    adjustments_made = True
                if self.lr_reduction_count > 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = min(
                            param_group['lr'] * 1.5, 
                            self.config['learning_rate']
                        )
                    print(f"  ⬆️  Increased learning rate")            
            if "TRAINING_DIVERGENCE" in issues:
                print(f"  ⚠️  Training divergence detected!")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"  ⬇️  Reduced learning rate to prevent divergence")            
            if "TRAINING_PLATEAU" in issues:
                print(f"  📉 Training plateau detected")
                if epoch % 5 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 1.2
                    print(f"  🔄 Increased LR to escape plateau")
            self.scheduler_cosine.step()
            self.scheduler_plateau.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < old_lr * 0.99:
                self.lr_reduction_count += 1
            self.lr_history.append(current_lr)
            self.wd_history.append(self.current_weight_decay)
            self.issue_history[epoch] = issues
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Train-Val Gap: {gap:.4f}")
            print(f"  LR:         {current_lr:.2e}")
            print(f"  Weight Decay: {self.current_weight_decay:.2e}")
            if issues:
                print(f"  Issues: {', '.join(issues)}")
            if adjustments_made:
                print(f"  ✅ Automatic adjustments applied")
            if epoch % 10 == 0:
                print(f"\n  📊 Detailed Metrics (Epoch {epoch}):")
                print(f"     Val MAE:  {val_mae:.4f} kcal/mol")
                print(f"     Val RMSE: {val_rmse:.4f} kcal/mol")
                print(f"     Val R²:   {val_r2:.4f}")
            if self.config['use_wandb']:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_val_gap': gap,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'learning_rate': current_lr,
                    'weight_decay': self.current_weight_decay,
                    'overfitting': 1 if 'OVERFITTING' in issues else 0,
                    'underfitting': 1 if 'UNDERFITTING' in issues else 0
                })
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss            
            if epoch % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, val_loss, is_best)
            if self.early_stopping(val_loss):
                if self.lr_reduction_count >= self.max_lr_reductions:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Max LR reductions ({self.max_lr_reductions}) reached")
                    break
                else:
                    print(f"\nWarning: Patience exhausted but attempting recovery...")
                    self.early_stopping.counter = 0
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print(f"Reduced LR to {self.optimizer.param_groups[0]['lr']:.2e}, continuing training...")
            torch.cuda.empty_cache()
            gc.collect()        
        print("\n" + "="*80)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Target RMSE < 1.5 kcal/mol for good performance")
        print("="*80)
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'lr_history': self.lr_history,
            'wd_history': self.wd_history,
            'issue_history': {str(k): v for k, v in self.issue_history.items()}
        }
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({k: v if isinstance(v, dict) else [float(x) for x in v] 
                      for k, v in history.items()}, f, indent=2)
def main():
    test_ic50_uM = 1.0  
    test_ic50_nM = test_ic50_uM * 1000
    test_deltaG = 1.36 * (np.log10(test_ic50_nM) - 9)
    print(f"\n{'='*80}")
    print(f"Conversion Test: IC50 = {test_ic50_uM} μM → ΔG = {test_deltaG:.2f} kcal/mol")
    print(f"Expected: ΔG ≈ -8.2 kcal/mol")
    print(f"{'='*80}\n")
    config = {
        'model': {
            'node_input_dim': 75,
            'edge_input_dim': 12,
            'hidden_dim': 256,
            'num_gin_layers': 5,
            'num_gat_layers': 2,
            'gat_heads': 8,
            'dropout': 0.15,
            'num_tasks': 1
        },
        'num_epochs': 25,
        'batch_size': 64,
        'learning_rate': 5e-5,
        'weight_decay': 1e-5,
        'grad_clip': 0.5,
        'min_lr': 1e-6,
        'scheduler_T0': 10,
        'scheduler_Tmult': 2,
        'patience': 20,
        'max_lr_reductions': 5,
        'save_every': 5,
        'use_amp': False,
        'use_wandb': False,
        'chembl_db_path': '/home/coder/project/data/chembl_36/chembl_36_sqlite/chembl_36.db',
        'cache_dir': '/home/coder/GNN/cache/chembl',
        'max_atoms': 150,
        'num_workers': 4,
        'output_dir': '/home/coder/GNN/outputs/chembl_pretrain',
        'seed': 42
    }
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    if config['use_wandb']:
        wandb.init(
            project='gnn-pic50-prediction',
            name=f'chembl_pretrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            config=config
        )
    print("Loading ChEMBL dataset...")
    dataset = ChEMBLDataset(
        db_path=config['chembl_db_path'],
        cache_dir=config['cache_dir'],
        max_atoms=config['max_atoms']
    )
    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train_split=0.9,
        val_split=0.1
    )    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Total molecules: {len(dataset):,}")
    trainer = ChEMBLPretrainer(config)
    trainer.train(train_loader, val_loader)
    config_path = Path(config['output_dir']) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)    
    if config['use_wandb']:
        wandb.finish()
if __name__ == '__main__':
    main()