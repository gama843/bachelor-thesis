import os
import torch

def train_and_validate(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, run_folder):
    """
    Handles training, validation, logging, and model saving.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to be trained and validated.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    test_loader : DataLoader
        DataLoader for the test dataset.
    criterion : torch.nn.Module
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer for model parameters.
    device : torch.device
        Device where the model is placed (CPU or CUDA).
    num_epochs : int
        Number of epochs for training.
    run_folder : str
        Path to the folder where model checkpoints and logs will be saved.

    Returns:
    --------
    None
    """
    os.makedirs(run_folder, exist_ok=True)
    model_path = os.path.join(run_folder, "model.pth")
    log_path = os.path.join(run_folder, "log.txt")

    with open(log_path, "w") as log_file:
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)

            log_msg = (f"Epoch {epoch+1}/{num_epochs} - "
                       f"Training Loss: {train_loss:.4f}, "
                       f"Validation Loss: {val_loss:.4f}, "
                       f"Validation Accuracy: {val_accuracy:.4f}\n")

            print(log_msg)
            log_file.write(log_msg)

            # save checkpoint
            checkpoint_path = os.path.join(run_folder, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

        # final test
        test_loss, test_accuracy = validate_one_epoch(model, test_loader, criterion, device)
        test_msg = f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n"
        print(test_msg)
        log_file.write(test_msg)

    # save final model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

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
        images = images.to(device)
        questions = torch.tensor(questions, dtype=torch.long, device=device)
        answers = torch.tensor(answers, dtype=torch.long, device=device)

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
            images = images.to(device)
            questions = torch.tensor(questions, dtype=torch.long, device=device)
            answers = torch.tensor(answers, dtype=torch.long, device=device)

            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += answers.size(0)
            correct += (predicted == answers).sum().item()

    accuracy = correct / total
    return running_loss / len(val_loader), accuracy