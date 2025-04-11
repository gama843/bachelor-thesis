import datetime
import matplotlib.pyplot as plt
import os

def get_question_type_and_subtype(question_vector):
    """
    Determine the type and subtype of a question from its one-hot encoded vector.

    Parameters
    ----------
    question_vector : list or np.ndarray
        One-hot encoded vector indicating the question type and subtype.

    Returns
    -------
    tuple
        A tuple (question_type, question_subtype), where question_type is either
        'relational' or 'non-relational', and question_subtype is one of:
        ['closest', 'farthest', 'count'] for relational,
        ['topbottom', 'leftright', 'shape'] for non-relational.

    Raises
    ------
    ValueError
        If the subtype is not recognized.
    """
    question_type = "relational" if question_vector[6] == 1 else "non-relational"

    subtype_map = {
        "relational": {
            8: "closest",
            9: "farthest",
            10: "count"
        },
        "non-relational": {
            8: "topbottom",
            9: "leftright",
            10: "shape"
        }
    }

    for index, subtype in subtype_map[question_type].items():
        if question_vector[index] == 1:
            return question_type, subtype

    raise ValueError(f"Unknown {question_type} question subtype")

def log_and_print(msg, file):
    print(msg)
    file.write(msg + "\n")

def get_experiment_name(num_images, model_type, image_form, question_form, seed, img_arch=None, note=""):
    today = datetime.datetime.now().strftime("%d%m%Y")

    if image_form == 'image':
        if not img_arch:
            raise ValueError("img_arch must be specified when image_form is 'image'")
        experiment_dir = f"experiments/{today}_{num_images}_{model_type}_{image_form}_{img_arch}_{question_form}_{seed}"
    else:
        experiment_dir = f"experiments/{today}_{num_images}_{model_type}_{image_form}_{question_form}_{seed}"

    if note:
        experiment_dir += f"_{note}"

    return experiment_dir

def plot_accuracy_breakdown(
    overall_accuracy,
    type_acc,
    subtype_acc,
    model_name,
    output_dir,
    relational_subtypes={"closest", "count", "farthest"}
):
    """
    Plot and save a horizontal bar chart summarizing accuracy breakdown.

    Parameters:
    -----------
    overall_accuracy : float
        Accuracy across all test samples.
    
    type_acc : dict
        Mapping from question types to their respective accuracies.
    
    subtype_acc : dict
        Mapping from question subtypes to their respective accuracies.
    
    model_name : str
        Name of the model (or 'Human' if plotting human data).

    output_dir : str
        Directory where the plot should be saved.

    relational_subtypes : set, optional
        Subtypes considered as relational (used for color coding).
    """

    labels = ["overall"] + list(type_acc.keys()) + list(subtype_acc.keys())
    accuracies = [overall_accuracy] + [type_acc[k] for k in type_acc] + [subtype_acc[k] for k in subtype_acc]
    colors = (
        ["tab:red"] +
        ["tab:orange"] * len(type_acc) +
        ["tab:blue" if label in relational_subtypes else "tab:green" for label in subtype_acc]
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels, accuracies, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Accuracy')
    ax.set_title(f'Performance overview ({model_name})')
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=0.8)

    for i, v in enumerate(accuracies):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"performance_overview_{model_name}.png")
    plt.savefig(plot_path)
    plt.close(fig)

    return plot_path

def plot_accuracy_breakdown(
    overall_accuracy,
    type_acc,
    subtype_acc,
    model_name,
    output_dir,
    relational_subtypes={"closest", "count", "farthest"}
):
    """
    Plot and save a horizontal bar chart summarizing accuracy breakdown.

    Parameters:
    -----------
    overall_accuracy : float
        Accuracy across all test samples.
    
    type_acc : dict
        Mapping from question types to their respective accuracies.
    
    subtype_acc : dict
        Mapping from question subtypes to their respective accuracies.
    
    model_name : str
        Name of the model (or 'Human' if plotting human data).

    output_dir : str
        Directory where the plot should be saved.

    relational_subtypes : set, optional
        Subtypes considered as relational (used for color coding).
    """
    import matplotlib.pyplot as plt
    import os

    label_order = [
        "topbottom", "shape", "leftright",
        "closest", "count", "farthest",
        "non-relational", "relational",
        "overall"
    ]

    full_acc = {"overall": overall_accuracy, **type_acc, **subtype_acc}

    labels = [label for label in label_order if label in full_acc]
    accuracies = [full_acc[label] for label in labels]

    colors = []
    for label in labels:
        if label == "overall":
            colors.append("tab:red")
        elif label in type_acc:
            colors.append("tab:orange")
        elif label in relational_subtypes:
            colors.append("tab:blue")
        else:
            colors.append("tab:green")

    labels = labels[::-1]
    accuracies = accuracies[::-1]
    colors = colors[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels, accuracies, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Accuracy')
    ax.set_title(f'Performance overview ({model_name})')
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=0.8)

    for i, v in enumerate(accuracies):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"performance_overview_{model_name}.png")
    plt.savefig(plot_path)
    plt.close(fig)

    return plot_path