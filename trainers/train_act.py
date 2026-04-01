"""
ACT-Branch训练
使用LogitNorm损失在轨迹数据集上训练
"""

import torch
from models.act_branch import LogitNormLoss


def train_act_branch(act_model, train_loader, device,
                     epochs=50, lr=1e-3, weight_decay=1e-4, tau=0.04):
    """
    训练ACT-Branch
    """
    act_model.to(device)
    act_model.train()

    criterion = LogitNormLoss(tau=tau)
    optimizer = torch.optim.AdamW(
        act_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for traj, labels in train_loader:
            traj, labels = traj.to(device), labels.to(device)
            logits = act_model(traj)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * traj.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += traj.size(0)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / total
            acc = correct / total * 100
            print(f"  Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

    return act_model