import torch
from monai.data import ThreadDataLoader
from monai.networks.nets import EfficientNetBN
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from src.optimizers import CosineAnnealingLRWarmup
from src.dataset import MyoblastDataset
from src.utils import create_transforms
from src.model_trainer import Trainer
from datetime import datetime
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")


EXPERIMENT="fibroblast_binary"
EPOCH=210
LOG_DIR = f"{EXPERIMENT}_{EPOCH}"
SAVE_PATH = f"experiments/{EXPERIMENT}/models"
#EXP_IDS = [211014, 220125, 220308, 220525]
#EXP_IDS = [220429, 220506]
EXP_IDS = [240221, 240219]

train_transforms, val_transforms, test_transforms = create_transforms()
train_dataset = MyoblastDataset(cell_type="fibroblast", exp_ids=EXP_IDS, mode="train", transform=train_transforms)
val_dataset = MyoblastDataset(cell_type="fibroblast", exp_ids=EXP_IDS, mode="val", transform=val_transforms)
# train_dataset = MyoblastDataset(cell_type="cancer", exp_ids=EXP_IDS, mode="train", transform=train_transforms)
# val_dataset = MyoblastDataset(cell_type="cancer", exp_ids=EXP_IDS, mode="val", transform=val_transforms)
train_loader = ThreadDataLoader(train_dataset, num_workers=4, batch_size=8, shuffle=True)
val_loader = ThreadDataLoader(val_dataset, num_workers=4, batch_size=8, shuffle=True)

labels = [sample['label'].item() for sample in train_dataset]
label_counts = Counter(labels)
print("Label count:", label_counts)
total_samples = sum(label_counts.values())
weights = [total_samples / (label_counts[i] * len(label_counts)) for i in label_counts.keys()]
weights_tensor = torch.tensor(weights, dtype=torch.half)
print("Weights:", weights)

NUM_CLASSES = len(label_counts)

model = EfficientNetBN("efficientnet-b7", pretrained=True, progress=False, spatial_dims=2, in_channels=1, num_classes=NUM_CLASSES)
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
scheduler = CosineAnnealingLRWarmup(optimizer, T_max=EPOCH, T_warmup=10)
scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter(log_dir=f"runs/{LOG_DIR}_{EXP_IDS}")
loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.00001)
trainer = Trainer(train_loader, val_loader, model, optimizer, scheduler, scaler, loss_fn, num_classes=NUM_CLASSES, device='cuda', save_path=f"experiments/{EXPERIMENT}/models", writer=writer, epochs=EPOCH)

trainer.train()
writer.close()