"""ResNet18在CIFAR-10上微调分类头
backbone冻结，只训练最后的全连接层
"""

import torch
import torch.nn.functional as F


def finetune(model, train_loader, device, epochs=10, lr=1e-3, weight_decay=1e-4):
    """
    微调分类头（backbone冻结）
    ResNet18的backbone已冻结，只训练classifier
    
    与DeiT版本的区别：
    - lr默认1e-3（ResNet分类头从零训练，需要更大学习率）
    - backbone的BN层保持eval模式
    """
    model.to(device)
    model.eval()  # backbone保持eval（BN层使用预训练统计量）
    model.classifier.train()  # 只有分类头进入训练模式

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # backbone冻结，不计算梯度
            with torch.no_grad():
                features = model.backbone_forward(x)  # [B, 512]
            
            logits = model.classifier(features)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        scheduler.step()
        avg_loss = total_loss / total
        acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

    return model
