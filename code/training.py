import os
import torch
import time
import numpy as np
from plotting import visualize_training_log
import progressbar
from utils import log_and_print

def train_and_validate(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, run_folder, question_form):
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

    total_start_time = time.time()

    with open(log_path, "w") as log_file:
        log_and_print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        
        epoch_bar = progressbar.ProgressBar(maxval=num_epochs,
                                         widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                 progressbar.Percentage(), ' ', 
                                                 'Training ', progressbar.ETA()])
        
        print('Training progress:')
        epoch_bar.start()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # training phase
            train_start_time = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, question_form)
            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            
            # validation phase
            val_start_time = time.time()
            val_loss, val_accuracy, breakdown, subtype_accuracy = validate_one_epoch(
                model, val_loader, criterion, device, question_form
            )
            val_end_time = time.time()
            val_duration = val_end_time - val_start_time
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_bar.update(epoch + 1)

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

        epoch_bar.finish()

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        log_and_print(f"\nTotal training time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)", log_file)
        
        log_and_print("\nEvaluating on test set...", log_file)
        test_start_time = time.time()
        test_loss, test_accuracy, test_breakdown, test_subtype_accuracy = validate_one_epoch(
            model, test_loader, criterion, device, question_form
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

def train_one_epoch(model, train_loader, criterion, optimizer, device, question_form):
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
    question_form : str
        Form of the question - can be 'binary' or 'string'.
    
    Returns:
    --------
    float
        Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    batch_bar = progressbar.ProgressBar(maxval=len(train_loader),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                              progressbar.Percentage(), ' ',
                                              'Epoch ', progressbar.ETA()])
    print('Epoch progress:')
    batch_bar.start()

    for batch_idx, (images, questions, binary_questions, answers, _, _) in enumerate(train_loader):
        images = images.to(device)
        if question_form == 'string':
            questions = torch.tensor(questions, dtype=torch.long, device=device)
        else:
            questions = binary_questions.to(device)
        answers = torch.tensor(answers, dtype=torch.long, device=device)

        optimizer.zero_grad()
        outputs = model(images, questions)
        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        batch_bar.update(batch_idx + 1)
    
    batch_bar.finish()
    return running_loss / len(train_loader)

def validate_one_epoch(model, val_loader, criterion, device, question_form):
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
        for images, questions, binary_questions, answers, q_types, q_subtypes in val_loader:
            images = images.to(device)
            questions = torch.tensor(questions, dtype=torch.long, device=device)
            binary_questions = binary_questions.to(device)
            answers = torch.tensor(answers, dtype=torch.long, device=device)
            
            if question_form == 'string':
                outputs = model(images, questions)
            elif question_form == 'binary':
                outputs = model(images, binary_questions)
            else:
                raise ValueError(f"Unsupported question form: {question_form}")
            
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

def save_train_answer_distribution(experiment_dir, dataset_builder):
    """
    Compute and save the answer distribution of the training set to a text file.

    Parameters
    ----------
    experiment_dir : str
        Path to the directory where the distribution file will be saved.

    dataset_builder : DatasetBuilder
        Instance that provides access to training samples and a method to compute answer distribution.
    """
    distribution = dataset_builder.compute_answer_distribution(dataset_builder.train_samples)
    output_path = os.path.join(experiment_dir, 'train_data_distribution.txt')
    
    with open(output_path, 'w') as f:
        for answer, count in distribution.items():
            f.write(f"{answer} {count}\n")

def load_answer_distribution(experiment_dir):
    """
    Load an answer distribution from a text file.

    Parameters
    ----------
    experiment_dir : str
        Path to the directory where the distribution file is located.

    Returns
    -------
    dict
        A dictionary mapping answers (str) to counts (int).
    """
    filename = 'train_data_distribution.txt'
    path = os.path.join(experiment_dir, filename)
    distribution = {}
    
    with open(path, 'r') as f:
        for line in f:
            answer, value = line.strip().split()
            distribution[answer] = float(value)
    
    return distribution

def compute_baseline_performance(test_loader, answer_vocab, experiment_dir):
    """
    Compute performance of three baselines:
    1. Random guessing from answer_vocab
    2. Most frequent class prediction
    3. Sampling from empirical distribution
    
    Parameters:
    -----------
    test_loader : DataLoader
        DataLoader for the test dataset.
    answer_vocab : dict
        Mapping of answer labels to indices.
    experiment_dir : str
        Directory where results will be saved.
    """
    
    all_answers = []
    all_question_types = []
    all_question_subtypes = []
    
    for _, _, _, answers, q_types, q_subtypes in test_loader:
        all_answers.extend(answers.numpy())
        all_question_types.extend(q_types)
        all_question_subtypes.extend(q_subtypes)
    
    all_answers = np.array(all_answers)
    all_question_types = np.array(all_question_types)
    all_question_subtypes = np.array(all_question_subtypes)
    
    unique_types = np.unique(all_question_types)
    unique_subtypes = np.unique(all_question_subtypes)
    relational_subtypes = ["furthest", "count", "closest"]
    
    probs = load_answer_distribution(experiment_dir)
    
    answer_to_idx = {}
    for answer, idx in answer_vocab.items():
        answer_to_idx[str(answer)] = idx
    
    answers_list = list(probs.keys())
    probs_list = [probs[answer] for answer in answers_list]
    
    most_frequent_answer = max(probs.items(), key=lambda x: x[1])[0]
    most_frequent_idx = answer_to_idx[most_frequent_answer]
    
    num_classes = len(answer_vocab)
    total_samples = len(all_answers)
    
    # predictions for each baseline
    random_preds = np.random.randint(0, num_classes, size=total_samples)
    most_freq_preds = np.full_like(all_answers, most_frequent_idx)
    
    indices = []
    dist = []
    
    for i, answer in enumerate(answers_list):
        indices.append(answer_to_idx[answer])
        dist.append(probs_list[i])
    
    sampled_answers = np.random.choice(
        indices,
        size=total_samples,
        p=dist
    )
    
    random_acc = (random_preds == all_answers).mean()
    most_freq_acc = (most_freq_preds == all_answers).mean()
    empirical_acc = (sampled_answers == all_answers).mean()
    
    log_path = os.path.join(experiment_dir, "baseline_performance.txt")
    
    with open(log_path, "w") as log_file:
        log_and_print(f"Baseline performance evaluation", log_file)
        log_and_print("\nOverall accuracy:", log_file)
        log_and_print(f"  Random guessing: {random_acc:.4f}", log_file)
        log_and_print(f"  Most frequent class: {most_freq_acc:.4f}", log_file)
        log_and_print(f"  Empirical distribution sampling: {empirical_acc:.4f}", log_file)
        
        # accuracy by question type
        log_and_print("\nAccuracy by question type:", log_file)
        for q_type in unique_types:
            type_indices = np.where(all_question_types == q_type)[0]
            type_random_acc = (random_preds[type_indices] == all_answers[type_indices]).mean()
            type_most_freq_acc = (most_freq_preds[type_indices] == all_answers[type_indices]).mean()
            type_empirical_acc = (sampled_answers[type_indices] == all_answers[type_indices]).mean()
            
            log_and_print(f"  {q_type}:", log_file)
            log_and_print(f"    Random guessing: {type_random_acc:.4f}", log_file)
            log_and_print(f"    Most frequent class: {type_most_freq_acc:.4f}", log_file)
            log_and_print(f"    Empirical distribution sampling: {type_empirical_acc:.4f}", log_file)
        
        # accuracy by question subtype
        log_and_print("\nAccuracy by question subtype:", log_file)
        for q_type in unique_types:
            log_and_print(f"  {q_type}:", log_file)
            
            # get subtypes for this question type
            if q_type == "relational":
                relevant_subtypes = [s for s in unique_subtypes if s in relational_subtypes]
            else:
                relevant_subtypes = [s for s in unique_subtypes if s not in relational_subtypes]
            
            for subtype in relevant_subtypes:
                subtype_indices = np.where(all_question_subtypes == subtype)[0]
                if len(subtype_indices) == 0:
                    continue
                    
                subtype_random_acc = (random_preds[subtype_indices] == all_answers[subtype_indices]).mean()
                subtype_most_freq_acc = (most_freq_preds[subtype_indices] == all_answers[subtype_indices]).mean()
                subtype_empirical_acc = (sampled_answers[subtype_indices] == all_answers[subtype_indices]).mean()
                
                log_and_print(f"    {subtype}:", log_file)
                log_and_print(f"      Random guessing: {subtype_random_acc:.4f}", log_file)
                log_and_print(f"      Most frequent class: {subtype_most_freq_acc:.4f}", log_file)
                log_and_print(f"      Empirical distribution sampling: {subtype_empirical_acc:.4f}", log_file)
    
    return {
        "random": random_acc,
        "most_frequent": most_freq_acc,
        "empirical": empirical_acc
    }