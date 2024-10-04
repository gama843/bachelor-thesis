import torch

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : DataLoader
        DataLoader for the training dataset.
    criterion : torch.nn.Module
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer for model parameters.
    device : torch.device
        Device where the model is placed (CPU or CUDA).
    
    Returns:
    --------
    float
        Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, questions, answers in train_loader:
        images, questions, answers = images.to(device), torch.tensor(questions).to(device), torch.tensor(answers).to(device)

        optimizer.zero_grad()

        outputs = model(images, questions)
        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def validate_one_epoch(model, val_loader, criterion, device):
    """
    Validate the model on the validation dataset.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to be validated.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : torch.nn.Module
        The loss function.
    device : torch.device
        Device where the model is placed (CPU or CUDA).
    
    Returns:
    --------
    float
        Average validation loss for the epoch.
    float
        Accuracy on the validation dataset.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, questions, answers in val_loader:
            images, questions, answers = images.to(device), torch.tensor(questions).to(device), torch.tensor(answers).to(device)

            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += answers.size(0)
            correct += (predicted == answers).sum().item()
    
    accuracy = correct / total
    return running_loss / len(val_loader), accuracy