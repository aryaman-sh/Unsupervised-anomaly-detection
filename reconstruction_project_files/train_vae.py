import torch
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from vae import ConvVAE, train_vae, valid_vae
from datasets import oasis_dataset

epochs = 300
batch_size = 64
spatial_size = 128
log_freq = 20
lr_rate = 1e-3
log_dir = '/home/Student/s4606685/24_1_22/logs'
device = torch.device('cuda')
model_name = 'one'

train_dataset = oasis_dataset(img_size = spatial_size, train=True, data_aug=1)
train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

vae_model = ConvVAE(spatial_size, model_name)
vae_model.double().to(device)

optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr_rate)
writer_train = SummaryWriter(log_dir + '/train_' + vae_model.name)
writer_valid = SummaryWriter(log_dir + '/valid_' + vae_model.name)


for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    loss, lat_loss, l2_loss, res_loss = train_vae(vae_model, train_data_loader, device, optimizer, epoch)
    print(("epoch %d: train_gen_loss %f train_lat_loss %f train_res_loss %f total train_loss %f") % (
        epoch, l2_loss, lat_loss, res_loss, loss))
    writer_train.add_scalar('total loss', loss, epoch)
    writer_train.add_scalar('l2 loss', l2_loss, epoch)
    writer_train.add_scalar('latent loss', lat_loss, epoch)
    writer_train.flush()

    # Save model
    if epoch % log_freq == 0:  # and not epoch == 0:
        vae_model.eval()
        lat_batch_sample = vae_model.sample(batch_size, device)
        writer_valid.add_image('Batch of sampled images', torch.clamp(lat_batch_sample, 0, 1), epoch,
                               dataformats='NCHW')

        img_test, mask = next(iter(train_data_loader))
        img_test = img_test.to(device)
        img_re, __, __, __ = vae_model(img_test.double())
        writer_valid.add_image('Batch of original images', img_test, epoch, dataformats='NCHW')
        writer_valid.add_image('Batch of reconstructed images', torch.clamp(img_re, 0, 1), epoch, dataformats='NCHW')

        path = log_dir + model_name + str(epoch) + '.pth'
        writer_valid.flush()
        torch.save(vae_model, path)
