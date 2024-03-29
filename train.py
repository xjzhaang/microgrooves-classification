import torch
from monai.data import ThreadDataLoader
from monai.networks.nets import EfficientNetBN, EfficientNetBNFeatures
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from src.optimizers import CosineAnnealingLRWarmup, StepLRWarmup
from src.dataset import MyoblastDataset
from src.utils import create_transforms, compute_mean_std
from src.model_trainer import Trainer
from datetime import datetime
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
import fnmatch


EXPERIMENT="cancer_multiclass"
EPOCH=120
LOG_DIR = f"{EXPERIMENT}_{EPOCH}"
SAVE_PATH = f"experiments/{EXPERIMENT}/models"
#EXP_IDS = [211014, 220125, 220308, 220525]
EXP_IDS = [220429, 220506]
EXP_IDS_VAL = [220508]
# EXP_IDS = [240215]
# EXP_IDS_VAL = [240219]
ds_dataset = MyoblastDataset(cell_type="cancer", exp_ids=EXP_IDS, mode="cropped", transform=None)
mean, std = compute_mean_std(ds_dataset)

print(f"Mean: {mean} | Std: {std}")
train_transforms, val_transforms, test_transforms = create_transforms(mean, std)
train_dataset = MyoblastDataset(cell_type="cancer", exp_ids=EXP_IDS, mode="cropped", transform=train_transforms)
val_dataset = MyoblastDataset(cell_type="cancer", exp_ids=EXP_IDS_VAL, mode="cropped", transform=val_transforms)
# train_dataset = MyoblastDataset(cell_type="cancer", exp_ids=EXP_IDS, mode="train", transform=train_transforms)
# val_dataset = MyoblastDataset(cell_type="cancer", exp_ids=EXP_IDS, mode="val", transform=val_transforms)
train_loader = ThreadDataLoader(train_dataset, num_workers=4, batch_size=16, shuffle=True)
val_loader = ThreadDataLoader(val_dataset, num_workers=4, batch_size=8, shuffle=True)

# Example usage:


labels = [sample['label'].item() for sample in train_dataset]
label_counts = Counter(labels)
print("Label count:", label_counts)
total_samples = sum(label_counts.values())
weights = [total_samples / (label_counts[i] * len(label_counts)) for i in label_counts.keys()]
weights_tensor = torch.tensor(weights, dtype=torch.half)
print("Weights:", weights)

labels = [sample['label'].item() for sample in val_dataset]
label_counts = Counter(labels)
print("Label count:", label_counts)
total_samples = sum(label_counts.values())
weights = [total_samples / (label_counts[i] * len(label_counts)) for i in label_counts.keys()]
weights_tensorv = torch.tensor(weights, dtype=torch.half)
print("Weights:", weights)


NUM_CLASSES = len(label_counts)

model = EfficientNetBN("efficientnet-b6", pretrained=True, progress=True, spatial_dims=2, in_channels=1, num_classes=NUM_CLASSES)
pattern = "_blocks.0.*"
pattern1 = "_blocks.1.*"
pattern3 = "_conv_stem*"
pattern2 = "_bn0*"
for name, param in model.named_parameters():
    if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name, pattern1) or fnmatch.fnmatch(name, pattern2) or fnmatch.fnmatch(name, pattern3):
        param.requires_grad = False
model._fc = torch.nn.Linear(in_features=2304, out_features=NUM_CLASSES, bias=True)

optimizer = torch.optim.AdamW(model.parameters(), 1e-4, eps=1e-05)
scheduler = StepLRWarmup(optimizer, T_max=EPOCH,  gamma=0.5, T_warmup=10)
scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter(log_dir=f"runs/{LOG_DIR}_{EXP_IDS}")
loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.00001)
loss_fn_val = torch.nn.CrossEntropyLoss(weight=weights_tensorv, label_smoothing=0.00001)

trainer = Trainer(train_loader, val_loader, model, optimizer, scheduler, scaler, loss_fn, loss_fn_val, num_classes=NUM_CLASSES, device='cuda', save_path=f"experiments/{EXPERIMENT}/models_{EXP_IDS}", writer=writer, epochs=EPOCH)

trainer.train()
writer.close()

