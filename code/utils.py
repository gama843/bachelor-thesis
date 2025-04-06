import datetime

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