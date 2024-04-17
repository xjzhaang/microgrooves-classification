import torch
from monai.data import ThreadDataLoader
from monai.networks.nets import EfficientNetBN, DenseNet121, ResNet, NetAdapter
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from src.optimizers import CosineAnnealingLRWarmup, StepLRWarmup
from src.dataset import MyoblastDataset
from src.utils import create_transforms, compute_mean_std
from src.model_trainer import Trainer
from datetime import datetime
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
import fnmatch


EXPERIMENT="fibroblast"
EPOCH=200
LOG_DIR = f"{EXPERIMENT}_{EPOCH}"
# EXP_IDS = [211007, 211014, 220125,220208]
# EXP_IDS_VAL = [ 210929,220201]
# EXP_IDS = [220429, 220508]
EXP_IDS = [240215, 240219]
EXP_IDS_VAL = [240221]
# EXP_IDS = [210929, 211007, 220201,220308, 220525]
# EXP_IDS_VAL = [211014, 220208, 220310, 220531]
ds_dataset = MyoblastDataset(cell_type="fibroblast", exp_ids=EXP_IDS, mode="cropped", transform=None)
mean, std = compute_mean_std(ds_dataset)
print(f"Mean: {mean} | Std: {std}")

SAVE_PATH = f"experiments/{EXPERIMENT}/models_{EXP_IDS}_{EPOCH}"


train_transforms, val_transforms, test_transforms = create_transforms(mean, std)
# train_dataset = MyoblastDataset(cell_type="myoblast", exp_ids=EXP_IDS, mode="train", transform=train_transforms)
# val_dataset = MyoblastDataset(cell_type="myoblast", exp_ids=EXP_IDS, mode="val", transform=val_transforms)
train_dataset = MyoblastDataset(cell_type="fibroblast", exp_ids=EXP_IDS, mode="train", transform=train_transforms)
val_dataset = MyoblastDataset(cell_type="fibroblast", exp_ids=EXP_IDS, mode="val", transform=val_transforms)
train_loader = ThreadDataLoader(train_dataset, num_workers=4, batch_size=16, shuffle=True)
val_loader = ThreadDataLoader(val_dataset, num_workers=4, batch_size=8, shuffle=True)


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

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
modified_conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

original_conv1_weights = model.conv1.weight
summed_weights = torch.sum(original_conv1_weights, dim=1, keepdim=True) 
modified_conv1.weight.data = summed_weights

model.conv1 = modified_conv1
model.fc = torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES, bias=True)

# freeze_layers_until = 'layer1' 
# freeze = True
# for name, param in model.named_parameters():
#     if freeze_layers_until in name:
#         freeze = False
#     if freeze:
#         param.requires_grad = False
        


optimizer = torch.optim.AdamW(model.parameters(), 1e-4, eps=1e-05)
scheduler = StepLRWarmup(optimizer, T_max=EPOCH,  gamma=0.1, T_warmup=1)
scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter(log_dir=f"runs/{LOG_DIR}_{EXP_IDS}")
loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.00001)
loss_fn_val = torch.nn.CrossEntropyLoss(weight=weights_tensorv, label_smoothing=0.00001)

trainer = Trainer(train_loader, val_loader, model, optimizer, scheduler, scaler, loss_fn, loss_fn_val, num_classes=NUM_CLASSES, device='cuda', save_path=SAVE_PATH, writer=writer, epochs=EPOCH)

trainer.train()
writer.close()

