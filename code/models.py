import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import onnx
import onnxruntime as ort
from torchvision.models import ResNet18_Weights

class ImageEncoder(nn.Module):
    """
    ImageEncoder class using a pre-trained ResNet18 model for feature extraction,
    with built-in image preprocessing.

    Methods:
    --------
    preprocess_image(image: torch.Tensor) -> torch.Tensor
        Applies preprocessing to an input image.
    forward(image: torch.Tensor) -> torch.Tensor
        Passes the preprocessed input image through the CNN to extract feature maps.
    """

    def __init__(self):
        super(ImageEncoder, self).__init__()
        # uses a pre-trained ResNet18 model, excluding the last fully connected layer
        self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize with ImageNet stats
        ])

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the input image by resizing and normalizing.

        Parameters:
        -----------
        image : torch.Tensor
            Input image tensor of shape (C, H, W).

        Returns:
        --------
        torch.Tensor
            Preprocessed image tensor of shape (1, 3, 224, 224), with a batch dimension.
        """
        preprocessed_image = self.preprocess(image)
        return preprocessed_image

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features from the input image.

        Parameters:
        -----------
        image : torch.Tensor
            Input image in tensor format.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, 512, H, W), where 512 is the number of feature
            channels, and H and W are the spatial dimensions of the output feature map.
        """
        preprocessed_image = self.preprocess_image(image)
        features = self.cnn(preprocessed_image)
        return features
    
class QuestionEncoder(nn.Module):
    """
    Module for encoding questions using an LSTM.


    Methods:
    -------
    forward(questions):
        Performs a forward pass through the network, returning the final hidden state of the LSTM.
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(QuestionEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, questions):
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        questions : torch.Tensor
            A batch of tokenized questions represented as tensors.

        Returns
        -------
        torch.Tensor
            The last hidden state of the LSTM for each question in the batch, with shape (batch_size, hidden_size).
        """
        
        embedded = self.embedding(questions)  # shape: (batch_size, seq_len, embed_size)
        _, (hidden, _) = self.lstm(embedded)  # only take the hidden state output

        # return the last hidden state of the LSTM
        return hidden[-1].unsqueeze(0)  # shape: (batch_size, hidden_size)
    
class RelationalNetwork(nn.Module):
    """
    Module for relational reasoning, composed of two MLPs (g_theta and f_phi).

    Attributes
    ----------
    g_theta : nn.Sequential
        An MLP that takes concatenated object features and question embeddings to learn relations.
    f_phi : nn.Sequential
        A final MLP that takes the summed output of all g_theta outputs to produce a final prediction.

    Methods
    -------
    forward(object_features, question_embedding):
        Performs a forward pass through the network using object features and a question embeddings.
    """

    def __init__(self, feature_dim, question_dim, g_theta_dim, f_phi_dim, num_classes):
        """
        Initializes the RelationalNetwork with two MLPs: g_theta and f_phi.

        Parameters
        ----------
        feature_dim : int
            The dimensionality of the object features.
        question_dim : int
            The dimensionality of the question embedding.
        g_theta_dim : int
            The dimensionality of the hidden layer in the g_theta MLP.
        f_phi_dim : int
            The dimensionality of the hidden layer in the f_phi MLP.
        num_classes : int
            The number of output classes for the multi-class classification.        
        """
        super(RelationalNetwork, self).__init__()
        
        self.g_theta = nn.Sequential(
            nn.Linear(2 * feature_dim + question_dim, g_theta_dim),
            nn.ReLU(),
            nn.Linear(g_theta_dim, g_theta_dim),
            nn.ReLU()
        )
        self.f_phi = nn.Sequential(
            nn.Linear(g_theta_dim, f_phi_dim),
            nn.ReLU(),
            nn.Linear(f_phi_dim, num_classes)
        )

    def forward(self, object_features, question_embedding):
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        object_features : torch.Tensor
            A tensor containing the features of objects, with shape (batch_size, num_objects, feature_dim).
        question_embedding : torch.Tensor
            A tensor representing the question embedding, with shape (batch_size, question_dim).

        Returns
        -------
        torch.Tensor
            The output tensor, which contains the prediction result. The shape is (batch_size, num_classes).
        """
        num_objects = object_features.size(1)
        relations = []
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    # concatenate the features of object pairs with the question embedding
                    pair_features = torch.cat(
                        [object_features[:, i], object_features[:, j], question_embedding], dim=1
                    )
                    relations.append(self.g_theta(pair_features))
        
        # sum the outputs from g_theta
        relations_sum = torch.stack(relations, dim=1).sum(dim=1)
        # apply f_phi to the summed output
        output = self.f_phi(relations_sum)
        
        return output
    
class RelationalReasoningModel(nn.Module):
    """
    Model for visual question answering using relational reasoning.

    This model consists of three main components:
    - ImageEncoder: encodes the input image into object features
    - QuestionEncoder: encodes the input question into a vector embedding
    - RelationalNetwork: performs relational reasoning over object features and the question embedding

    Methods
    -------
    forward(image, question):
        Performs a forward pass through the network, processing the image and question to produce an output.
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, feature_dim, g_theta_dim, f_phi_dim, num_classes):
        """
        Initializes the RelationalReasoningModel with an ImageEncoder, QuestionEncoder, and RelationalNetwork.

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary used in the QuestionEncoder.
        embed_size : int
            The dimensionality of the word embeddings in the QuestionEncoder.
        hidden_size : int
            The number of features in the hidden state of the LSTM in the QuestionEncoder.
        num_layers : int
            The number of recurrent layers in the LSTM of the QuestionEncoder.
        feature_dim : int
            The dimensionality of the object features produced by the ImageEncoder.
        g_theta_dim : int
            The dimensionality of the hidden layers in the g_theta MLP of the RelationalNetwork.
        f_phi_dim : int
            The dimensionality of the hidden layers in the f_phi MLP of the RelationalNetwork.
        num_classes : int
            The number of output classes for the multi-class classification.
        """
        super(RelationalReasoningModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.question_encoder = QuestionEncoder(vocab_size, embed_size, hidden_size, num_layers)
        self.relation_network = RelationalNetwork(feature_dim, hidden_size, g_theta_dim, f_phi_dim, num_classes)
    
    def forward(self, image, question):
        """
        Performs a forward pass through the model.

        Parameters
        ----------
        image : torch.Tensor
            A tensor representing the input image of shape (batch_size, channels, height, width).
        question : torch.Tensor
            A tensor representing the input question of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            The output tensor, representing the predicted class logits, with shape (batch_size, num_classes).
        """
        image_features = self.image_encoder(image)
        # flatten the spatial dimensions to produce a set of object features
        object_features = image_features.view(image_features.size(0), -1, image_features.size(1))
        question_embedding = self.question_encoder(question)
        # perform relational reasoning using the object features and question embedding
        output = self.relation_network(object_features, question_embedding)
        
        return output
    
class BaselineModel(nn.Module):
    """
    Base model that combines a CNN-based image encoder and an LSTM-based question encoder
    with a simple MLP for classification without a relational network module.
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super(BaselineModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.question_encoder = QuestionEncoder(vocab_size, embed_size, hidden_size, num_layers)
        
        # MLP for classification
        self.fc1 = nn.Linear(512 * 7 * 7 + hidden_size, 256)  # where 512 is the number of channels in the output feature map of ResNet18 and 7x7 are the spatial dimensions, hidden_size comes from the dimensionality of the question embeddings
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, image: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get the model's prediction.
        
        Parameters:
        -----------
        image : torch.Tensor
            Input image in tensor format.
        questions : torch.Tensor
            Input tensor containing word indices of shape (batch_size, seq_len).

        Returns:
        --------
        torch.Tensor
            Output prediction tensor of shape (batch_size, num_classes).
        """
        
        # encode the image and flatten the features
        image_features = self.image_encoder(image)  # shape: (batch_size, 512, 7, 7)
        
        image_features = image_features.view(image_features.size(0), -1)  # flatten to (batch_size, 512*7*7)
        
        # encode the question
        question_embedding = self.question_encoder(questions)  # shape: (batch_size, hidden_size)
        
        # concatenate image and question features
        combined_features = torch.cat((image_features, question_embedding), dim=1)  # shape: (batch_size, 512*7*7 + hidden_size)
        
        # pass through the MLP
        x = nn.ReLU()(self.fc1(combined_features))
        output = self.fc2(x)
        
        return output

# This module uses the @public annotation to include certain private methods in the generated documentation.
# The @public tag is applied only for the purpose of documentation generation.

class ModelConstructor:
    """
    A class to manage and load different types of models: predefined models (BaselineModel, RelationalReasoningModel) 
    or custom models in ONNX or PyTorch format, please mind the ONNX can be used only for inference.

    Methods:
    --------
    load_model(model_type: str, **kwargs) -> nn.Module:
        Loads a predefined model ('baseline', 'relational') or a custom ONNX/PyTorch model based on the input parameters.
    """

    def __init__(self):
        pass

    def load_model(self, model_type: str, **kwargs) -> torch.nn.Module:
        """
        Load and initialize one of the predefined models or a custom model.

        Parameters:
        -----------
        model_type : str
            The type of model to load ('baseline', 'relational', 'onnx', or 'custom').
        **kwargs : dict
            Additional arguments required to initialize the model (e.g., vocab_size, embed_size, etc.).

        Returns:
        --------
        nn.Module or ort.InferenceSession
            The initialized PyTorch model or ONNX session for inference.
        """
        if model_type == 'baseline':
            return BaselineModel(
                vocab_size=kwargs['vocab_size'],
                embed_size=kwargs['embed_size'],
                hidden_size=kwargs['hidden_size'],
                num_layers=kwargs['num_layers'],
                num_classes=kwargs['num_classes']
            )
        elif model_type == 'relational':
            return RelationalReasoningModel(
                vocab_size=kwargs['vocab_size'],
                embed_size=kwargs['embed_size'],
                hidden_size=kwargs['hidden_size'],
                num_layers=kwargs['num_layers'],
                feature_dim=kwargs['feature_dim'],
                g_theta_dim=kwargs['g_theta_dim'],
                f_phi_dim=kwargs['f_phi_dim'],
                num_classes=kwargs['num_classes']
            )
        elif model_type == 'onnx':
            return self._load_custom_model_onnx(kwargs['onnx_path'])
        elif model_type == 'custom':
            return self._load_custom_model(kwargs['model_class'], kwargs.get('weights_path'))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _load_custom_model(self, model_class: torch.nn.Module, weights_path: str = None) -> torch.nn.Module:
        """
        @public
        
        Loads a custom PyTorch model and optionally loads pre-trained weights.

        Parameters:
        -----------
        model_class : torch.nn.Module
            The class or instance of the custom model to load.
        weights_path : str, optional
            Path to the PyTorch model weights file (.pt or .pth) (default=None).

        Returns:
        --------
        nn.Module
            The initialized PyTorch model, optionally with loaded weights.
        """
        model = model_class() if isinstance(model_class, type) else model_class

        if weights_path:
            model.load_state_dict(torch.load(weights_path))

        return model

    def _load_custom_model_onnx(self, onnx_path: str):
        """
        @public

        Loads a custom model from an ONNX file using ONNX Runtime.

        Parameters:
        -----------
        onnx_path : str
            Path to the ONNX model file.

        Returns:
        --------
        ort.InferenceSession
            An ONNX Runtime session that can be used to make predictions with the ONNX model.
        """
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        session = ort.InferenceSession(onnx_path)

        return session