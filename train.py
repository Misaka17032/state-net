import torch
from state import States
from torchvision import datasets, transforms
import os

batch_size = 128
epoches = 100
save_round = 50

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
train_datasets = datasets.ImageFolder("./data/train", transform = train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = batch_size, shuffle = True)

states = States()
for epoch in range(epoches):
	print("epoch", epoch)
	for index, (datax, datay) in enumerate(train_dataloader):
		if (index + 1) % 10 == 0:
			states.eval(datax.numpy(), datay.numpy())
		else:
			states.train(datax.numpy(), datay.numpy())
	if (epoch + 1) % save_round == 0:
		states.save()