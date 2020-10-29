import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import sys
#sns.set()

class CA(nn.Module):
    def __init__(self, ca_size=(100, 100), n_channels=16, fire_rate=0.5, living_threshold=0.1):
        super(CA, self).__init__()
        self.n_channels = n_channels
        self.fire_rate = fire_rate
        self.ca_size = ca_size
        self.living_threshold = living_threshold
        self.dx = torch.from_numpy(np.asarray([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])).view(1,1,3,3).repeat(n_channels, n_channels, 1, 1).float()
        self.dy = torch.from_numpy(np.asarray([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]).T).view(1,1,3,3).repeat(n_channels, n_channels, 1, 1).float()
        self.identity = torch.from_numpy(np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]])).view(1,1,3,3).repeat(n_channels, n_channels, 1, 1).float()
        self.conv1 = nn.Conv2d(48, 128, 1)
        self.conv2 = nn.Conv2d(128, n_channels, 1)
        self.conv2.weight.data.fill_(0)
        self.conv2.bias.data.fill_(0)

    def get_living(self, x):
        alpha = x[:, 3, :, :]
        return nn.functional.max_pool2d(alpha, 3, 1, 1) > self.living_threshold

    def forward(self, x):
        pre_update = self.get_living(x)
        dx = nn.functional.conv2d(x, self.dx.to(next(self.parameters()).device), padding=1)
        dy = nn.functional.conv2d(x, self.dy.to(next(self.parameters()).device), padding=1)
        ident = nn.functional.conv2d(x, self.identity.to(next(self.parameters()).device), padding=1)
        y = torch.cat((dx, dy, ident), 1)
        y = self.conv1(y)
        y = torch.nn.functional.leaky_relu(y)
        y = self.conv2(y)

        rand_mask = (torch.rand(x.size(0), *self.ca_size) > self.fire_rate).view(x.size(0), 1, *self.ca_size).repeat(1, x.size(1), 1, 1).to(next(self.parameters()).device)
        x = x+y*rand_mask
        post_update = self.get_living(x)
        alive = (pre_update*post_update).unsqueeze(1).repeat(1, x.size(1), 1, 1)
        return x*alive

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
n_batches = 4

model = CA().to(device)
orig_state = np.zeros((1, 16, 100, 100))
orig_state[:, :, 100//2, 100//2] = 1.0
plt.imshow(orig_state[0, :3, :, :].transpose((1,2,0)))
plt.savefig('seed.png')
plt.close()
orig_state = torch.from_numpy(orig_state).float().to(device)

optim = torch.optim.Adam(model.parameters(), lr=0.002)

img_np = Image.open('shiba.png').resize((100, 100))
img_np = np.array(img_np)/255.

plt.imshow(img_np[:, :, :3])
plt.savefig('target.png')
plt.close()
img = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0).float().to(device).repeat(n_batches, 1, 1, 1)

pool = None
n_epochs = 8000
state = None
for epoch in range(n_epochs):
    optim.zero_grad()

    if state is None:
        state = orig_state.detach().repeat(n_batches, 1, 1, 1)

    for _ in range(np.random.randint(64, 96)):
        state = model(state)

    worst_idx = torch.argmax(torch.mean(torch.nn.functional.mse_loss(state[:, :3, :, :], img[:, :3, :, :], reduction='none'), dim=(1,2,3)))
    loss = torch.nn.functional.mse_loss(state[:, :3, :, :], img[:, :3, :, :]) #+ torch.nn.functional.l1_loss(state[:, :3, :, :], torch.zeros_like(state[:, :3, :, :]))

    loss.backward()
    for p in model.parameters():
        p.grad = p.grad/(torch.norm(p.grad)+1e-8)
    optim.step()

    sys.stdout.write('\rEpoch: {}\tLoss: {:.4}'.format(epoch, loss.item()))
    sys.stdout.flush()

    if epoch%10 == 0:
        img_new = state.detach().cpu().numpy()[:, :3, :, :].reshape(n_batches, 3, 100, 100).transpose((0, 2, 3, 1))
        fig = plt.figure(figsize=(8, 8))
        for i in range(1, int(np.sqrt(n_batches))**2 + 1):
            fig.add_subplot(int(np.sqrt(n_batches)), int(np.sqrt(n_batches)), i)
            plt.imshow(img_new[i-1])
        plt.savefig('progress.png')
        plt.close()

    state = state.detach()
    state[worst_idx] = orig_state.detach()