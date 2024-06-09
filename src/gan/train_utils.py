import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: DataLoader,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        criterion: nn.Module,
        device="cpu"
):
    generator.to(device)
    discriminator.to(device)

    generator.train()
    discriminator.train()

    total_loss_generator = 0.0
    total_loss_discriminator = 0.0
    total_output_d_real = 0.0
    total_output_d_fake = 0.0

    for step, batch in enumerate(train_loader):
        batch_size = batch.size(0)
        batch = batch.to(device)

        # Optimize generator
        optimizer_g.zero_grad()
        targets = torch.ones(batch_size).to(device)  # Generator is rewarded for fooling the discriminator
        fake_music = generator.generate(batch_size, device)
        outputs_fake = discriminator(fake_music)
        loss_g = criterion(outputs_fake.reshape(-1), targets)
        loss_g.backward()
        total_loss_generator += loss_g.item()
        optimizer_g.step()

        # Optimize discriminator for real data
        optimizer_d.zero_grad()
        # targets = torch.ones(batch_size).to(device)  # Real data
        targets = 0.8 + 0.4 * torch.rand(batch_size).to(device)  # Label Smoothing
        outputs_real = discriminator(batch)
        total_output_d_real += outputs_real.mean().item()
        loss_d_real = criterion(outputs_real.reshape(-1), targets)
        loss_d_real.backward()

        # Optimize discriminator for fake data
        # targets = torch.zeros(batch_size).to(device)  # Fake data
        targets = 0.15 * torch.rand(batch_size).to(device)  # Label Smoothing
        fake_music = generator.generate(batch_size, device)
        outputs_fake = discriminator(fake_music)
        total_output_d_fake += outputs_fake.mean().item()
        loss_d_fake = criterion(outputs_fake.reshape(-1), targets)
        loss_d_fake.backward()
        total_loss_discriminator += (loss_d_real + loss_d_fake).item()
        optimizer_d.step()

        # print progress
        if step % 10 == 9:
            print(".", end="")
        if step % 100 == 99:
            print("|", end="")
    print()

    return (
        total_loss_generator / len(train_loader),
        total_loss_discriminator / len(train_loader) / 2,
        total_output_d_real / len(train_loader),
        total_output_d_fake / len(train_loader),
    )


def eval_one_epoch(
        generator: nn.Module,
        discriminator: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device="cpu"
):
    generator.to(device)
    discriminator.to(device)

    generator.eval()
    discriminator.eval()

    total_loss_generator = 0.0
    total_loss_discriminator = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch_size = batch.size(0)

            fake_music = generator.generate(batch_size, device)

            # Optimize generator
            targets = torch.ones(batch_size).to(device)  # Generator is rewarded for fooling the discriminator
            outputs = discriminator(fake_music)
            loss = criterion(outputs.reshape(-1), targets)
            total_loss_generator += loss.item()

            # Optimize discriminator for real data
            inputs = batch.to(device)
            targets = torch.ones(batch_size).to(device)  # Real data
            outputs = discriminator(inputs)
            loss = criterion(outputs.reshape(-1), targets)
            total_loss_discriminator += loss.item()

            # Optimize discriminator for fake data
            targets = torch.zeros(batch_size).to(device)  # Fake data
            outputs = discriminator(fake_music)
            loss = criterion(outputs.reshape(-1), targets)
            total_loss_discriminator += loss.item()

        return total_loss_generator / len(test_loader), total_loss_discriminator / len(test_loader) / 2
