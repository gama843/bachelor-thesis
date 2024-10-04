import os
import json
import random
from collections import Counter

from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DatasetBuilder:
    def __init__(self, data_dir, transform=None, transform_prob=0, random_seed=42):
        """
        Initializes the DatasetBuilder by loading, splitting, and processing the dataset.

        Parameters:
        -----------
        data_dir : str
            The directory containing the images and dataset JSON file.
        transform : torchvision.transforms.Compose
            Transformations to apply to the images.
        transform_prob : float
            The probability of applying transformations (default=0).
        random_seed : int
            Random seed for reproducibility (default=42).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.transform_prob = transform_prob
        self.random_seed = random_seed
        self._load_data()

        # compute vocab and max_len based on the training set
        self.vocab = self._build_vocab(self.train_samples)
        self.answer_vocab = self._build_answer_vocab(self.train_samples)
        self.max_len = self._compute_max_question_len(self.train_samples)

        # create the individual datasets
        self.train_dataset = RelationalDataset(self.train_samples, self.vocab, self.answer_vocab, self.max_len, transform=self.transform, transform_prob=self.transform_prob, random_seed=self.random_seed)
        self.val_dataset = RelationalDataset(self.val_samples, self.vocab, self.answer_vocab, self.max_len, transform=self.transform, transform_prob=self.transform_prob, random_seed=self.random_seed)
        self.test_dataset = RelationalDataset(self.test_samples, self.vocab, self.answer_vocab, self.max_len, transform=self.transform, transform_prob=self.transform_prob, random_seed=self.random_seed)

    def _load_data(self, train_size=0.7, val_size=0.1, test_size=0.2):
        """
        @public
        
        Loads the dataset from the 'descr.json' file and splits it into train, validation, and test sets.

        Parameters:
        -----------
        train_size : float
            Proportion of the data to be used for training.
        val_size : float
            Proportion of the data to be used for validation.
        test_size : float
            Proportion of the data to be used for testing.
        """
        total_size = train_size + val_size + test_size
        if not abs(total_size - 1.0) < 1e-6:
            raise ValueError(f"The sum of train_size ({train_size}), val_size ({val_size}), and test_size ({test_size}) must be 1.0")

        with open(os.path.join(self.data_dir, 'descr.json'), 'r') as f:
            data = json.load(f)

        img_paths = list(data.keys())

        train_val_paths, test_paths = train_test_split(img_paths, test_size=test_size, random_state=self.random_seed)
        val_relative_size = val_size / (train_size + val_size)
        train_paths, val_paths = train_test_split(train_val_paths, test_size=val_relative_size, random_state=self.random_seed)

        self.train_samples = self._generate_samples(data, train_paths)
        self.val_samples = self._generate_samples(data, val_paths)
        self.test_samples = self._generate_samples(data, test_paths)

    def _generate_samples(self, data, image_paths):
        """
        @public

        Generates samples for the dataset (20 per image: 10 relational, 10 non-relational).

        Parameters:
        -----------
        data : dict
            The loaded dataset dictionary.
        image_paths : list
            A list of image paths corresponding to a specific split.

        Returns:
        --------
        list
            A list of samples where each sample is a tuple (image_path, question, answer).
        """
        samples = []
        for image_path in image_paths:
            img_info = data[image_path]
            for obj_info in img_info:
                if 'question' in obj_info:
                    samples.append((os.path.join(self.data_dir, image_path), obj_info['question'], obj_info['answer']))
        return samples

    def _compute_max_question_len(self, samples):
        """
        @public

        Computes the maximum length of questions in the dataset.

        Parameters:
        -----------
        samples : list
            A list of samples where each sample contains a question.

        Returns:
        --------
        int
            The length of the longest question in the dataset.
        """
        max_len = 0
        for _, question, _ in samples:
            question_len = len(question.split())
            max_len = max(max_len, question_len)
        return max_len

    def _build_vocab(self, samples, min_freq=1):
        """
        @public

        Builds a vocabulary from the dataset questions and assigns an index to each unique word.

        Parameters:
        -----------
        samples : list
            A list of samples where each sample contains a question.
        min_freq : int
            Minimum frequency for a word to be included in the vocabulary (default=1).

        Returns:
        --------
        dict
            A dictionary mapping each word to a unique index.
        """
        word_counter = Counter()

        for _, question, _ in samples:
            word_counter.update(question.lower().split())

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, freq in word_counter.items():
            if freq >= min_freq:
                vocab[word] = len(vocab)

        return vocab

    def _build_answer_vocab(self, samples):
        """
        @public

        Builds a vocabulary from the dataset answers, mapping each answer to a unique index.

        Parameters:
        -----------
        samples : list
            A list of samples where each sample contains an answer.

        Returns:
        --------
        dict
            A dictionary mapping each answer to a unique index.
        """
        answer_counter = Counter()

        for _, _, answer in samples:
            answer_counter.update([answer])

        answer_vocab = {ans: idx for idx, ans in enumerate(answer_counter.keys())}

        return answer_vocab
    
class RelationalDataset(Dataset):
    """
    A PyTorch Dataset class to load relational reasoning data, including both relational 
    and non-relational questions. Supports data augmentation and train/val/test splitting.

    Attributes:
    -----------
    samples : list
        The dataset samples with image paths, questions, and answers.
    vocab : dict
        A dictionary mapping question words to token indices.
    answer_vocab : dict
        A dictionary mapping answers to token indices.
    max_len : int
        The maximum length of tokenized questions for padding.
    transform : ImageAnswerTransform
        Data transformations that apply both image and text transformations.
    transform_prob : float
        The probability of applying transformations.        
    random_seed : int
        Random seed for reproducibility and for use in transformations.
    """

    def __init__(self, samples, vocab, answer_vocab, max_len, transform=None, transform_prob=0, random_seed=42):
        self.samples = samples
        self.vocab = vocab
        self.answer_vocab = answer_vocab
        self.max_len = max_len
        self.transform = transform
        self.transform_prob = transform_prob        
        self.random_seed = random_seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetch an image and its corresponding tokenized question-answer pair, applying the appropriate transformations.

        Returns:
        --------
        tuple
            A tuple (image, tokenized_question, encoded_answer) where image is a transformed image tensor, 
            tokenized_question is a list of token indices, and encoded_answer is an integer index.
        """
        image_path, question, answer = self.samples[idx]
        img = Image.open(image_path)

        if self.transform and random.random() < self.transform_prob:
            img, answer = self.transform(img, answer)

        img = transforms.ToTensor()(img)
        tokenized_question = self._tokenize_question(question)
        encoded_answer = self._encode_answer(answer)

        return img, tokenized_question, encoded_answer

    def _tokenize_question(self, question):
        """
        @public

        Tokenizes the question into a list of token indices and pads or truncates to max_len.

        Parameters:
        -----------
        question : str
            The question to tokenize.

        Returns:
        --------
        list
            A list of token indices.
        """
        tokens = [self.vocab.get(word, self.vocab["<UNK>"]) for word in question.lower().split()]
        tokens = tokens[:self.max_len]  # truncate if longer than max_len
        tokens += [self.vocab["<PAD>"]] * (self.max_len - len(tokens))  # pad if shorter than max_len
        return tokens

    def _encode_answer(self, answer):
        """
        @public

        Converts an answer into its corresponding index.

        Parameters:
        -----------
        answer : str
            The answer to encode.
        answer_vocab : dict
            A dictionary mapping answers to indices.

        Returns:
        --------
        int
            The index of the answer in the answer vocabulary.
        """
        return self.answer_vocab[answer]

class ImageAnswerTransform:
    """
    A class that applies both image and text transformations, allowing 
    them to communicate via a shared state. The state can be used to 
    pass transformation parameters or results from the image transformation 
    to the text transformation, allowing for coordinated changes between 
    the image and the label.

    Attributes:
    -----------
    image_transform : callable, optional
        A transformation function for the image. It should accept a PIL Image 
        and a state dictionary, then return the transformed PIL Image and the updated state.
        
    text_transform : callable, optional
        A transformation function for the label (string). It should accept a string label 
        and the shared state, then return the transformed label.
        
    state : dict, optional
        A shared state dictionary used to pass information between the image 
        and text transformations. If not provided, it is initialized as an empty dictionary.

    Methods:
    --------
    __call__(img, label)
        Applies the image transformation, updates the shared state, and then 
        applies the text transformation using the updated state.

        Parameters:
        -----------
        img : PIL.Image
            The input image to be transformed.
            
        label : str
            The label associated with the image, which may be transformed 
            based on the shared state.

        Returns:
        --------
        img : PIL.Image
            The transformed image.

        label : str
            The transformed label, modified based on the image transformation 
            and the state.
    """

    def __init__(self, image_transform=None, text_transform=None, state=None):
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.state = state or {}

    def __call__(self, img, label):
        if self.image_transform:
            img, self.state = self.image_transform(img, self.state)

        if self.text_transform:
            label = self.text_transform(label, self.state)

        return img, label

class Rotate180DegreesTransform(ImageAnswerTransform):
    """
    A specialized ImageAnswerTransform that rotates the image by 180 degrees and modifies the answer accordingly.

    This class inherits from ImageAnswerTransform and implements the specific functionality of rotating an image
    by 180 degrees and transforming the directional answers ('top', 'bottom', 'left', 'right') based on the rotation.

    Methods:
    --------
    rotate_image_180(img, state):
        Rotates the image by 180 degrees clockwise.
    modify_answer_based_on_rotation(answer, state):
        Modifies the answer according to the 180-degree rotation.
    """

    def __init__(self, state=None):
        """
        Initializes the Rotate180DegreesTransform with predefined image and text transformations.

        Parameters:
        -----------
        state : dict, optional
            A shared state dictionary used to pass information between the image and text transformations.
            If not provided, it is initialized as an empty dictionary.
        """
        super().__init__(
            image_transform=self.rotate_image_180,
            text_transform=self.modify_answer_based_on_rotation,
            state=state
        )

    def rotate_image_180(self, img, state):
        """
        Rotate the image by 180 degrees clockwise.

        Parameters:
        -----------
        img : PIL.Image
            The image to be rotated.
        state : dict
            The state used for shared information between image and text transformations.

        Returns:
        --------
        img : PIL.Image
            The rotated image.
        state : dict
            The unchanged state dictionary.
        """
        img = img.rotate(-180)  # 180 degrees clockwise
        return img, state

    def modify_answer_based_on_rotation(self, answer, state):
        """
        Modify the answer based on the 180-degree rotation, assuming directional answers ('top', 'bottom', 'left', 'right').

        Parameters:
        -----------
        answer : str
            The original answer related to the image.
        state : dict
            The state used for shared information between image and text transformations.

        Returns:
        --------
        str
            The modified answer after applying the 180-degree rotation.
        """
        if answer in ['top', 'bottom', 'left', 'right']:
            if answer == 'top':
                return 'bottom'
            elif answer == 'bottom':
                return 'top'
            elif answer == 'left':
                return 'right'
            elif answer == 'right':
                return 'left'
        
        return answer