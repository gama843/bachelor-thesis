import os
import torch
import time
import numpy as np
from plotting import visualize_training_log

def train_and_validate(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, run_folder):
    """
    Handles training, validation, logging, and model saving with metrics matching the paper.

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
        Number of training epochs.
    run_folder : str
        Directory where model checkpoints and logs will be saved.
    answer_vocab : dict
        Mapping of answer labels to indices for decoding class labels.
    """

    os.makedirs(run_folder, exist_ok=True)
    model_path = os.path.join(run_folder, "model.pth")
    log_path = os.path.join(run_folder, "log.txt")

    def log_and_print(msg, file):
        print(msg)
        file.write(msg + "\n")

    total_start_time = time.time()

    with open(log_path, "w") as log_file:
        log_and_print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # training phase
            train_start_time = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            
            # validation phase
            val_start_time = time.time()
            val_loss, val_accuracy, breakdown, subtype_accuracy = validate_one_epoch(
                model, val_loader, criterion, device
            )
            val_end_time = time.time()
            val_duration = val_end_time - val_start_time
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            log_and_print(f"\nEpoch {epoch+1}/{num_epochs}", log_file)
            log_and_print(f"Epoch duration: {epoch_duration:.2f} seconds", log_file)
            log_and_print(f"  Training time: {train_duration:.2f} seconds", log_file)
            log_and_print(f"  Validation time: {val_duration:.2f} seconds", log_file)
            
            log_and_print(f"Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}", log_file)

            log_and_print("\nAccuracy breakdown per question type:", log_file)
            for category, acc in breakdown.items():
                log_and_print(f"  {category}: {acc:.4f}", log_file)

            log_and_print("\nAccuracy breakdown per question subtype:", log_file)
            for q_type, subtypes in subtype_accuracy.items():
                log_and_print(f"  {q_type}:", log_file)
                for subtype, acc in subtypes.items():
                    log_and_print(f"    {subtype}: {acc:.4f}", log_file)

            checkpoint_path = os.path.join(run_folder, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        log_and_print(f"\nTotal training time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)", log_file)
        
        log_and_print("\nEvaluating on test set...", log_file)
        test_start_time = time.time()
        test_loss, test_accuracy, test_breakdown, test_subtype_accuracy = validate_one_epoch(
            model, test_loader, criterion, device
        )
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        log_and_print(f"Test evaluation completed in {test_duration:.2f} seconds", log_file)

        # log test results with the same metrics as the paper
        log_and_print("\nTest set performance", log_file)
        log_and_print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}", log_file)

        # accuracy breakdown by question type
        log_and_print("\nTest accuracy breakdown per question type:", log_file)
        for category, acc in test_breakdown.items():
            log_and_print(f"  {category}: {acc:.4f}", log_file)

        # accuracy breakdown by question subtype
        log_and_print("\nTest accuracy breakdown per question subtype:", log_file)
        for q_type, subtypes in test_subtype_accuracy.items():
            log_and_print(f"  {q_type}:", log_file)
            for subtype, acc in subtypes.items():
                log_and_print(f"    {subtype}: {acc:.4f}", log_file)

        torch.save(model.state_dict(), model_path)
        log_and_print(f"Final model saved to {model_path}", log_file)
        
        log_and_print(f"\nTraining and evaluation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file)

    with open(log_path, "a") as log_file:
        # generate visualizations
        plots_dir = os.path.join(run_folder, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        log_and_print(f"\nGenerating training visualizations in {plots_dir}", log_file)
        
        vis_start_time = time.time()
        visualize_training_log(log_path, plots_dir)
        vis_end_time = time.time()
        vis_duration = vis_end_time - vis_start_time
        log_and_print(f"Visualization generation completed in {vis_duration:.2f} seconds", log_file)

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

    for images, questions, answers, _, _ in train_loader:
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
    Validate the model on the validation dataset with metrics matching the original paper.

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
        Overall accuracy on the validation dataset.
    dict
        Accuracy breakdown per question type (relational vs. non-relational).
    dict
        Accuracy breakdown per question subtype.
    """

    model.eval()
    running_loss = 0.0
    all_subtypes = ["furthest", "count", "closest", "shape", "topbottom", "leftright"]
    relational_subtypes = ["furthest", "count", "closest"]
    # non_relational_subtypes = ["shape", "topbottom", "leftright"]
    
    all_predictions = []
    all_true_labels = []
    all_question_types = []
    all_question_subtypes = []

    with torch.no_grad():
        for images, questions, answers, q_types, q_subtypes in val_loader:
            images = images.to(device)
            questions = torch.tensor(questions, dtype=torch.long, device=device)
            answers = torch.tensor(answers, dtype=torch.long, device=device)
            
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(answers.cpu().numpy())
            all_question_types.extend(q_types)
            all_question_subtypes.extend(q_subtypes)
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_question_types = np.array(all_question_types)
    all_question_subtypes = np.array(all_question_subtypes)
    
    # overall accuracy (primary metric used in the paper)
    total_samples = len(all_true_labels)
    correct_predictions = (all_predictions == all_true_labels).sum()
    overall_accuracy = correct_predictions / total_samples
    
    # question type breakdown (relational vs. non-relational)
    # this matches the paper's comparison between relational and non-relational questions
    breakdown_accuracy = {}
    for q_type in ["relational", "non-relational"]:
        type_indices = np.where(all_question_types == q_type)[0]
        if len(type_indices) > 0:
            type_accuracy = (all_predictions[type_indices] == all_true_labels[type_indices]).mean()
            breakdown_accuracy[q_type] = float(type_accuracy)
    
    # question subtype breakdown (furthest, count, closest, shape, etc.)
    # this matches the paper's detailed breakdown by question categories
    subtype_accuracy = {"relational": {}, "non-relational": {}}
    for subtype in all_subtypes:
        subtype_indices = np.where(all_question_subtypes == subtype)[0]
        if len(subtype_indices) > 0:
            subtype_accuracy_value = (all_predictions[subtype_indices] == all_true_labels[subtype_indices]).mean()
            parent_type = "relational" if subtype in relational_subtypes else "non-relational"
            subtype_accuracy[parent_type][subtype] = float(subtype_accuracy_value)

    return (
        running_loss / len(val_loader),
        float(overall_accuracy),
        breakdown_accuracy,
        subtype_accuracy
    )