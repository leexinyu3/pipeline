from face_network import InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

torch.cuda.empty_cache()
data_dir = 'C:/Users/hp/Desktop/DongChengdong/Data/training'
batch_size = 8
epochs = 25
workers = 0

save_checkpoints_path = './checkpoints/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

resnet = InceptionResnetV1(
    classify=True,
    pretrained=None,
    num_classes=201  #num_classes = the number of identities in training dataset. For e.g., if the training dataset has 300 identities, you must set num_classes to 300
).to(device)


optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(data_dir, transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.9 * len(img_inds))]
val_inds = img_inds[int(0.9 * len(img_inds)):]
train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

loss_fn = torch.nn.CrossEntropyLoss()

metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}


writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    if epoch%4 == 0:
        torch.save({
            'inceptionNet': resnet.state_dict()
        },save_checkpoints_path+'inceptionNet'+'%04d.pth'%epoch
        )

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

writer.close()