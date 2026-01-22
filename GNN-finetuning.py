import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import json
import gc
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from GNNmodel import create_model
from GNNdataset import PDBBindDataset, create_dataloaders
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, mode='min'):
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
    def reset(self):
        self.counter = 0
        self.early_stop = False
class PDBBindFineTuner:
    def __init__(self, config, pretrained_path=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print("Initializing model...")
        self.model = create_model(config['model']).to(self.device)
        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
            model_dict = self.model.state_dict()
            pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
            if config.get('freeze_backbone', False):
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if not k.startswith('predictor')}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(pretrained_dict)} layers from pretrained weights")
            if config.get('freeze_backbone', False):
                print("Freezing backbone layers...")
                for name, param in self.model.named_parameters():
                    if not name.startswith('predictor'):
                        param.requires_grad = False
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {num_params:,} / {total_params:,}")
        self.criterion = nn.MSELoss()
        self.current_weight_decay = config.get('weight_decay', 1e-6)
        self.initial_weight_decay = config.get('weight_decay', 1e-6)
        self._setup_optimizer()
        self._setup_scheduler()
        self.use_amp = config.get('use_amp', False) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 50),
            min_delta=0.001,
            mode='min'
        )        
        self.best_val_rmse = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_val_gap = []
        self.lr_reduction_count = 0
        self.max_lr_reductions = config.get('max_lr_reductions', 5)
        self.lr_history = []
        self.wd_history = []
        self.issue_history = {}
        self.metrics_history = {
            'train_rmse': [], 'val_rmse': [],
            'train_r2': [], 'val_r2': [],
            'train_pearson': [], 'val_pearson': [],
            'train_spearman': [], 'val_spearman': []
        }        
    def _setup_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.get('learning_rate', 1e-3),
            weight_decay=self.current_weight_decay
        )        
    def _setup_scheduler(self):
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.get('scheduler_patience', 20),
            min_lr=self.config.get('min_lr', 1e-7)
        )    
    def calculate_metrics(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)        
        if len(y_true) < 2 or len(y_pred) < 2:
            return {
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'r2': 0.0,
                'pearson': 0.0,
                'spearman': 0.0
            }        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        try:
            r2 = r2_score(y_true, y_pred)
            if np.isnan(r2) or np.isinf(r2):
                r2 = 0.0
        except Exception:
            r2 = 0.0
        try:
            pearson_r, _ = pearsonr(y_true, y_pred)
            if np.isnan(pearson_r):
                pearson_r = 0.0
        except Exception:
            pearson_r = 0.0
        try:
            spearman_r, _ = spearmanr(y_true, y_pred)
            if np.isnan(spearman_r):
                spearman_r = 0.0
        except Exception:
            spearman_r = 0.0        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'pearson': pearson_r,
            'spearman': spearman_r
        }    
    def adjust_weight_decay(self, train_loss, val_loss, epoch):
        if epoch % 5 != 0:
            return False            
        gap = train_loss - val_loss
        if gap < -0.8 and self.current_weight_decay < 0.01:
            self.current_weight_decay *= 2.0  
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = self.current_weight_decay
            print(f"  ⚠️  Overfitting detected! Increased weight decay to {self.current_weight_decay:.2e}")
            return True        
        return False    
    def unfreeze_layers_gradually(self, epoch):
        if not self.config.get('gradual_unfreezing', False):
            return False        
        warmup = self.config.get('warmup_epochs', 20)
        if epoch == warmup:
            print(f"  🔓 Phase 2: Unfreezing top layers (GAT)")
            for name, param in self.model.named_parameters():
                if 'gat' in name.lower() or 'attention' in name.lower():
                    param.requires_grad = True
            self._reset_optimizer_and_scheduler()
            return True        
        elif epoch == warmup * 2:
            print(f"  🔓 Phase 3: Unfreezing middle layers (GIN 3-5)")
            for name, param in self.model.named_parameters():
                if 'gin' in name.lower():
                    if any(f'gin.{i}' in name or f'gin_layers.{i}' in name for i in [2, 3, 4]):
                        param.requires_grad = True
            self._reset_optimizer_and_scheduler()
            return True        
        elif epoch == warmup * 3:
            print(f"  🔓 Phase 4: Unfreezing all layers (full fine-tuning)")
            for param in self.model.parameters():
                param.requires_grad = True
            self._reset_optimizer_and_scheduler()
            return True        
        return False    
    def _reset_optimizer_and_scheduler(self):
        current_lr = self.optimizer.param_groups[0]['lr']
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]  
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=current_lr,
            weight_decay=self.current_weight_decay
        )        
        self._setup_scheduler()        
        print(f"    Optimizer reset with {len(trainable_params)} parameter groups at LR={current_lr:.2e}")    
    def detect_training_issues(self, train_loss, val_loss, epoch):
        issues = []
        gap = train_loss - val_loss
        if gap < -0.8:
            issues.append("SEVERE_OVERFITTING")
        if epoch > 5 and len(self.train_losses) >= 5:
            recent_train = self.train_losses[-5:]
            if all(recent_train[i] < recent_train[i+1] for i in range(len(recent_train)-1)):
                issues.append("TRAINING_DIVERGENCE")        
        return issues    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_targets = []        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue            
            try:
                batch = batch.to(self.device)              
                if torch.isnan(batch.y).any() or torch.isinf(batch.y).any():
                    continue                
                self.optimizer.zero_grad()                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(batch).squeeze()
                        pred = torch.clamp(pred, min=-25.0, max=5.0)
                        loss = self.criterion(pred, batch.y.squeeze())
                else:
                    pred = self.model(batch).squeeze()
                    pred = torch.clamp(pred, min=-25.0, max=5.0)
                    loss = self.criterion(pred, batch.y.squeeze())              
                l2_reg = 0.01 * torch.mean(pred ** 2)
                loss = loss + l2_reg                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Warning: NaN/Inf loss detected in batch {batch_idx}, skipping")
                    continue                
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print(f"  Warning: NaN/Inf predictions in batch {batch_idx}, skipping")
                    continue                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.get('grad_clip', 1.0)
                    )                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"  Warning: Invalid gradients in batch {batch_idx}, skipping")
                        continue                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.get('grad_clip', 1.0)
                    )                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"  Warning: Invalid gradients in batch {batch_idx}, skipping")
                        continue                    
                    self.optimizer.step()                
                total_loss += loss.item()
                num_batches += 1                
                all_preds.extend(pred.detach().cpu().numpy().tolist())
                all_targets.extend(batch.y.squeeze().cpu().numpy().tolist()) 
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                if batch_idx % 50 == 0 and batch_idx > 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Warning: OOM in batch {batch_idx}, clearing cache and skipping")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    print(f"  Error in batch {batch_idx}: {e}")
                    continue
            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue
        if num_batches == 0:
            print("  Warning: No valid batches in training epoch")
            return float('inf'), self.calculate_metrics([0], [0])
        avg_loss = total_loss / num_batches
        metrics = self.calculate_metrics(all_targets, all_preds)
        return avg_loss, metrics    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_targets = []        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
            for batch in pbar:
                if batch is None:
                    continue                
                try:
                    batch = batch.to(self.device)
                    if torch.isnan(batch.y).any() or torch.isinf(batch.y).any():
                        continue                    
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            pred = self.model(batch).squeeze()
                            pred = torch.clamp(pred, min=-25.0, max=5.0)
                            loss = self.criterion(pred, batch.y.squeeze())
                    else:
                        pred = self.model(batch).squeeze()
                        pred = torch.clamp(pred, min=-25.0, max=5.0)
                        loss = self.criterion(pred, batch.y.squeeze())
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue                    
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        continue                    
                    total_loss += loss.item()
                    num_batches += 1                    
                    all_preds.extend(pred.cpu().numpy().tolist())
                    all_targets.extend(batch.y.squeeze().cpu().numpy().tolist())                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})                    
                except Exception as e:
                    print(f"  Error in validation batch: {e}")
                    continue        
        if num_batches == 0:
            print("  Warning: No valid batches in validation epoch")
            return float('inf'), self.calculate_metrics([0], [0])        
        avg_loss = total_loss / num_batches
        metrics = self.calculate_metrics(all_targets, all_preds)        
        return avg_loss, metrics    
    def test(self, test_loader):
        print("\n" + "="*80)
        print("Testing on test set...")
        print("="*80)        
        self.model.eval()
        all_preds = []
        all_targets = []        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            for batch in pbar:
                if batch is None:
                    continue                
                try:
                    batch = batch.to(self.device)
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            pred = self.model(batch).squeeze()
                            pred = torch.clamp(pred, min=-25.0, max=5.0)
                    else:
                        pred = self.model(batch).squeeze()
                        pred = torch.clamp(pred, min=-25.0, max=5.0)                    
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(batch.y.squeeze().cpu().numpy())
                except Exception as e:
                    print(f"  Error in test batch: {e}")
                    continue
        if len(all_preds) == 0:
            print("  Error: No valid test predictions")
            return {}, [], []
        metrics = self.calculate_metrics(all_targets, all_preds)
        print("\nTest Set Results:")
        print("-" * 80)
        print(f"MAE:      {metrics['mae']:.4f} kcal/mol")
        print(f"RMSE:     {metrics['rmse']:.4f} kcal/mol")
        print(f"R²:       {metrics['r2']:.4f}")
        print(f"Pearson:  {metrics['pearson']:.4f} (Goal: > 0.8)")
        print(f"Spearman: {metrics['spearman']:.4f}")
        print("="*80)
        return metrics, all_preds, all_targets
    def save_checkpoint(self, epoch, val_rmse, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_rmse': val_rmse,
            'metrics': metrics,
            'config': self.config,
            'current_weight_decay': self.current_weight_decay,
            'lr_reduction_count': self.lr_reduction_count
        }
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✅ Saved best model with RMSE: {val_rmse:.4f}")
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()
    def train(self, train_loader, val_loader):
        print("\n" + "="*80)
        print("Starting PDBBind Fine-tuning (ΔG Prediction)")
        print("Target: ΔG in kcal/mol from PDBBind native labels")
        print("Goal: Pearson R > 0.8 on core set")
        print("="*80)
        for epoch in range(1, self.config.get('num_epochs', 500) + 1):
            print(f"\nEpoch {epoch}/{self.config.get('num_epochs', 500)}")
            print("-" * 80)
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            val_loss, val_metrics = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)
            gap = train_loss - val_loss
            self.train_val_gap.append(gap)
            issues = self.detect_training_issues(train_loss, val_loss, epoch)
            adjustments_made = False
            if self.unfreeze_layers_gradually(epoch):
                adjustments_made = True
            if "SEVERE_OVERFITTING" in issues:
                print(f"  🔴 Severe overfitting detected (gap: {gap:.4f})")
                if self.adjust_weight_decay(train_loss, val_loss, epoch):
                    adjustments_made = True
            if "TRAINING_DIVERGENCE" in issues:
                print(f"  ⚠️  Training divergence detected!")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"  ⬇️  Reduced learning rate to prevent divergence")
                adjustments_made = True
                self.early_stopping.reset()
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < old_lr * 0.99:
                self.lr_reduction_count += 1
                print(f"  ℹ️  Scheduler reduced LR (reduction {self.lr_reduction_count}/{self.max_lr_reductions})")
            self.lr_history.append(current_lr)
            self.wd_history.append(self.current_weight_decay)
            self.issue_history[epoch] = issues
            self.metrics_history['train_rmse'].append(train_metrics['rmse'])
            self.metrics_history['val_rmse'].append(val_metrics['rmse'])
            self.metrics_history['train_r2'].append(train_metrics['r2'])
            self.metrics_history['val_r2'].append(val_metrics['r2'])
            self.metrics_history['train_pearson'].append(train_metrics['pearson'])
            self.metrics_history['val_pearson'].append(val_metrics['pearson'])
            self.metrics_history['train_spearman'].append(train_metrics['spearman'])
            self.metrics_history['val_spearman'].append(val_metrics['spearman'])
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - Loss: {train_loss:.4f}, RMSE: {train_metrics['rmse']:.4f} kcal/mol, "
                  f"R²: {train_metrics['r2']:.4f}, Pearson: {train_metrics['pearson']:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, RMSE: {val_metrics['rmse']:.4f} kcal/mol, "
                  f"R²: {val_metrics['r2']:.4f}, Pearson: {val_metrics['pearson']:.4f}")
            print(f"  Train-Val Gap: {gap:.4f}")
            print(f"  LR: {current_lr:.2e}, Weight Decay: {self.current_weight_decay:.2e}")
            if issues:
                print(f"  Issues: {', '.join(issues)}")
            if adjustments_made:
                print(f"  ✅ Automatic adjustments applied")
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_rmse': train_metrics['rmse'],
                    'train_r2': train_metrics['r2'],
                    'train_pearson': train_metrics['pearson'],
                    'val_loss': val_loss,
                    'val_rmse': val_metrics['rmse'],
                    'val_r2': val_metrics['r2'],
                    'val_pearson': val_metrics['pearson'],
                    'train_val_gap': gap,
                    'learning_rate': current_lr,
                    'weight_decay': self.current_weight_decay,
                    'severe_overfitting': 1 if 'SEVERE_OVERFITTING' in issues else 0,
                    'training_divergence': 1 if 'TRAINING_DIVERGENCE' in issues else 0
                })
            is_best = val_metrics['rmse'] < self.best_val_rmse
            if is_best:
                self.best_val_rmse = val_metrics['rmse']
            if epoch % self.config.get('save_every', 5) == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics['rmse'], val_metrics, is_best)
            if self.early_stopping(val_loss):
                if self.lr_reduction_count >= self.max_lr_reductions:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                    print(f"   Max LR reductions ({self.max_lr_reductions}) reached")
                    break
                else:
                    print(f"\n⚠️  Patience exhausted but attempting recovery...")
                    self.early_stopping.reset()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print(f"   Reduced LR to {self.optimizer.param_groups[0]['lr']:.2e}, continuing training...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        print("\n" + "="*80)
        print("Training completed!")
        print(f"Best validation RMSE: {self.best_val_rmse:.4f} kcal/mol")
        print("="*80)
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_val_gap': self.train_val_gap,
            'lr_history': self.lr_history,
            'wd_history': self.wd_history,
            'issue_history': {str(k): v for k, v in self.issue_history.items()},
            'metrics_history': self.metrics_history
        }
        history_path = self.output_dir / '/home/coder/GNN/outputs/pdbbind_finetune/training_history.json'
        with open(history_path, 'w') as f:
            json.dump({k: v if isinstance(v, dict) else [float(x) if not isinstance(x, list) else x for x in v] 
                      for k, v in history.items()}, f, indent=2)
def main():
    config = {
        'model': {
            'node_input_dim': 75,
            'edge_input_dim': 12,
            'hidden_dim': 256,
            'num_gin_layers': 5,
            'num_gat_layers': 2,
            'gat_heads': 8,
            'dropout': 0.3,  
            'num_tasks': 1
        },
        'num_epochs': 500,
        'batch_size': 32,
        'learning_rate': 5e-4,  
        'weight_decay': 1e-4, 
        'grad_clip': 0.5,  
        'min_lr': 1e-7,
        'scheduler_patience': 10,  
        'patience': 30,  
        'max_lr_reductions': 5,
        'save_every': 5,
        'use_amp': False,
        'use_wandb': False,
        'freeze_backbone': False,
        'gradual_unfreezing': False,
        'warmup_epochs': 15,
        'pdbbind_dir': '/home/coder/project/data/refined-set',
        'index_file': '/home/coder/project/data/refined-set/index/INDEX_refined_data.2020',
        'cache_dir': '/home/coder/GNN/cache/pdbbind_refined',
        'max_atoms': 150,
        'num_workers': 4,
        'pretrained_path': '/home/coder/GNN/outputs/chembl_pretrain/best_model.pt',
        'output_dir': '/home/coder/GNN/outputs/pdbbind_finetune',
        'seed': 42
    }
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if config.get('use_wandb', False):
        wandb.init(
            project='gnn-pic50-prediction',
            name=f'pdbbind_finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            config=config
        )
    print("Loading PDBBind dataset...")
    try:
        dataset = PDBBindDataset(
            data_dir=config['pdbbind_dir'],
            index_file=config['index_file'],
            cache_dir=config['cache_dir'],
            max_atoms=config['max_atoms']
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    print("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            train_split=0.8,
            val_split=0.1
        )
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        sys.exit(1)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    try:
        trainer = PDBBindFineTuner(config, pretrained_path=config.get('pretrained_path'))
        trainer.train(train_loader, val_loader)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    best_model_path = Path(config['output_dir']) / 'best_model.pt'
    if best_model_path.exists():
        try:
            print("\nLoading best model for testing...")
            checkpoint = torch.load(best_model_path, weights_only=False)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            test_metrics, preds, targets = trainer.test(test_loader)
            results_path = Path(config['output_dir']) / 'test_predictions.npz'
            np.savez(results_path, predictions=preds, targets=targets)
            print(f"Saved test predictions to {results_path}")
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Warning: No best model found, skipping test evaluation")
    config_path = Path(config['output_dir']) / 'config.json'
    with open(config_path, 'w') as f:
        json_config = {}
        for k, v in config.items():
            if isinstance(v, Path):
                json_config[k] = str(v)
            else:
                json_config[k] = v
        json.dump(json_config, f, indent=4)    
    if config.get('use_wandb', False):
        wandb.finish()    
    print("\n✅ Training pipeline completed successfully!")
if __name__ == '__main__':
    main()