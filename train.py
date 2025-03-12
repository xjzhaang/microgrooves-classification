import torch
from monai.data import ThreadDataLoader
from monai.networks.nets import EfficientNetBN, DenseNet121, ResNet, NetAdapter
from torchvision.models import resnet50, ResNet50_Weights, resnet34
from src.resnets import FFResNet50, FFResNet34
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from src.optimizers import CosineAnnealingLRWarmup, StepLRWarmup
from src.dataset import MyoblastDataset
from src.utils import create_transforms, compute_mean_std
from src.model_trainer import Trainer
from src.utils import set_deterministic_mode
from src_contrastive.model_contrastive import SimCLR
from datetime import datetime
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
import fnmatch
from pathlib import Path


EXPERIMENT="laminac_fibroblast_scaled"
EPOCH=100

# EXP_IDS = [210929, 220125, 220201, 220308] #, 220531,]#
# EXP_IDS_VAL = [211014, 220208, 211007,220310]  #220525,] #] #

EXP_IDS = [240215, 240219, 240805, 240807,] 
EXP_IDS_VAL = [240221, 240731, 240802]
# EXP_IDS = [240221, 240219, 240731, 240802] 
# EXP_IDS_VAL = [240215, 240805, 240807]
# FOLD1 = [240215, 240805, ] 
# FOLD2 = [240219, 240802,240807]
# FOLD3 = [240221, 240731]
# EXP_IDS = [240219, 240802,240807, 240221, 240731] 
# EXP_IDS_VAL = [240215, 240805,]

# FOLD1 = [210929, 220525] 
# FOLD2 = [211007, 220201, 220531]
# FOLD3 = [211014, 220208, 220614]

# EXP_IDS = [210929, 220525, 220201, 211014, 220308 ]
# EXP_IDS_VAL = [211007,  220531, 220208, 220310] 

ds_dataset = MyoblastDataset(cell_type=EXPERIMENT, exp_ids=EXP_IDS, mode="train", transform=None)
mean, std = compute_mean_std(ds_dataset)
print(f"Mean: {mean} | Std: {std}")

LOG_DIR = f"{EXPERIMENT}_{EXP_IDS}_{EPOCH}"
SAVE_PATH = f"experiments/{EXPERIMENT}/{EXP_IDS}_{EPOCH}"

train_transforms, val_transforms, test_transforms = create_transforms(mean, std)
train_dataset = MyoblastDataset(cell_type=EXPERIMENT, exp_ids=EXP_IDS, mode="train", transform=train_transforms)
val_dataset = MyoblastDataset(cell_type=EXPERIMENT, exp_ids=EXP_IDS, mode="val_exp", transform=val_transforms)
train_loader = ThreadDataLoader(train_dataset, num_workers=2, batch_size=16, shuffle=True)

def worker_init_fn(worker_id):
    set_deterministic_mode(42 + worker_id)
    
val_loader = ThreadDataLoader(val_dataset, num_workers=2, batch_size=1, shuffle=False, worker_init_fn=worker_init_fn)


def compute_class_weights(train_loader):
    label_counts = Counter()
    # Efficient batch processing instead of loading sample by sample
    for batch in train_loader:
        labels = batch['label'].cpu().numpy()  # Assuming 'label' is in the batch
        label_counts.update(labels)            # Update counts for this batch

    label_counts = dict(sorted(label_counts.items()))
    print("Label count:", label_counts)
    
    total_samples = sum(label_counts.values())
    num_classes = len(label_counts)
    
    # Compute weights for each class
    weights = [total_samples / (label_counts[i] * num_classes) for i in label_counts.keys()]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    return weights_tensor
    
weights_tensor = compute_class_weights(train_loader)
print("Class Weights:", weights_tensor)
weights_tensorv = compute_class_weights(val_loader)
print("Class Weights val:", weights_tensorv)


NUM_CLASSES = len(weights_tensorv)


### TRAIN MULTI-CLASS USING BINARY WEIGHTS

#model = FFResNet50()#weights=ResNet50_Weights.IMAGENET1K_V2
#model = FFResNet34()

# model.fc = torch.nn.Sequential(
#     torch.nn.Dropout(p=0.2), 
#     torch.nn.Linear(in_features=2048, out_features=2, bias=True)
# )

# *parts, last_part = Path(SAVE_PATH).parts
# last_part_name = last_part.split("_")[0] + "_100"
# print(last_part_name)
# binary_save_path = Path(*parts) / "resnet50" / "binary" / last_part_name

# checkpoint = torch.load(binary_save_path / "best_loss2.pth", map_location="cuda")
# model.load_state_dict(checkpoint['model'])
# print("binary model loaded!")

# model.fc = torch.nn.Sequential(
#     torch.nn.Dropout(p=0.2), 
#     torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES, bias=True)
# )

### TRAIN MULTI-CLASS USING CONTRASTIVE WEIGHTS

backbone = FFResNet50()
backbone.fc = torch.nn.Identity()
pretrained_ssl = SimCLR(
    backbone,
    input_dim=2048, 
    hidden_dim=2048,
    output_dim=128
)

checkpoint = torch.load(f"./pretrain/{EXPERIMENT}/resnet50-no-solarize/{EXP_IDS}_200/best_loss.pth", map_location="cuda")
pretrained_ssl.load_state_dict(checkpoint['model'])

# Extract just the backbone from the SimCLR model
model = pretrained_ssl.backbone
print("SSL model loaded!")

model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2), 
    torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES, bias=True)
)

unfreeze_after = ['initial', 'conv1', 'layer1', 'ff_parser_1', 'layer2', 'ff_parser_2', 'layer3', 'ff_parser_3', 'ff_parser_4',]#'layer4' 'ff_parser_5',] # Add your layer names here

freeze = True
for name, param in model.named_parameters():
    # Check if current layer name starts with any of the specified names
    should_unfreeze = any(name.startswith(layer) for layer in unfreeze_after)
    if should_unfreeze:
        freeze = False
    param.requires_grad = not freeze

        

optimizer = torch.optim.AdamW(model.parameters(), 0.0001)#, eps=1e-05)
scheduler = StepLRWarmup(optimizer, T_max=EPOCH,  gamma=0.5, T_warmup=0)
#scheduler = CosineAnnealingLRWarmup(optimizer, T_max=EPOCH,  T_warmup=5)
scaler = torch.cuda.amp.GradScaler(init_scale=2**14,)

writer = SummaryWriter(log_dir=f"runs/{LOG_DIR}")
loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.00001)
loss_fn_val = torch.nn.CrossEntropyLoss(weight=weights_tensorv, label_smoothing=0.00001) #weight=weights_tensorv, 

trainer = Trainer(train_loader, val_loader, model, optimizer, scheduler, scaler, loss_fn, loss_fn_val, num_classes=NUM_CLASSES, device='cuda', save_path=SAVE_PATH, writer=writer, epochs=EPOCH)

trainer.train()
writer.close()

