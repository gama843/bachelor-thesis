import torch
import torch.nn as nn
from torchvision import models, transforms
import onnx
import onnxruntime as ort
from torchvision.models import ResNet18_Weights

class CNNImageEncoder(nn.Module):
    """
    CNN-based image encoder for Sort-of-CLEVR dataset as described in the paper
    'A simple neural network module for relational reasoning'.
    
    Uses 4 convolutional layers with 32, 64, 128, and 256 kernels respectively,
    with ReLU activations and batch normalization.
    
    Methods:
    --------
    forward(image: torch.Tensor) -> torch.Tensor
        Processes the input image through the CNN to extract feature maps.
    """
    def __init__(self, input_channels=3):
        super(CNNImageEncoder, self).__init__()
        
        # 4 convolutional layers as described in the paper
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
    
    def forward(self, image):
            """
            Forward pass to extract features from the input image and add coordinate information.
            
            Parameters:
            -----------
            image : torch.Tensor
                Input image tensor of shape (batch_size, channels, height, width).
                For Sort-of-CLEVR, typically (batch_size, 3, 75, 75).
                
            Returns:
            --------
            objects : torch.Tensor
                Output tensor with each spatial location as an "object" with content
                features and coordinate information.
                Shape: (batch_size, d*d, 256+2) where d*d is the number of cells
                in the final feature map.
            """
            # features 
            feature_maps = self.conv_layers(image)
            batch_size, channels, height, width = feature_maps.shape
            
            # tagging with normalized spatial coords 

            # 1D tensor [height] -> [height, 1] -> [height, 1] * width 
            y_coords = torch.linspace(-1, 1, height).unsqueeze(1).expand(height, width)

            # 1D tensor [width] -> [height, width]
            x_coords = torch.linspace(-1, 1, width).expand(height, width)
            
            # reshape coords to match feature map dims
            coords = torch.stack((y_coords, x_coords), dim=0)  # [2, height, width]
            coords = coords.unsqueeze(0).expand(batch_size, 2, height, width)
            
            # concat feature maps and coordinates along the channel dimension
            feature_maps_with_coords = torch.cat([feature_maps, coords.to(feature_maps.device)], dim=1)
            
            # [batch_size, channels+2, height, width] -> [batch_size, height*width = objects, channels+2 = features + coords]
            # [batch_size, channels+2, height, width] -> [batch_size, height*width, channels+2]
            objects = feature_maps_with_coords.permute(0, 2, 3, 1).reshape(batch_size, height*width, channels+2)
            
            return objects
    
class ResNetImageEncoder(nn.Module):
    """
    ImageEncoder class using a pre-trained ResNet18 model for feature extraction,
    with built-in image preprocessing. Doesn't use spatial coordinates of objects as the CNNImageEncoder.

    Methods:
    --------
    preprocess_image(image: torch.Tensor) -> torch.Tensor
        Applies preprocessing to an input image.
    forward(image: torch.Tensor) -> torch.Tensor
        Passes the preprocessed input image through the CNN to extract feature maps.
    """

    def __init__(self, return_object_features=False):
        super(ResNetImageEncoder, self).__init__()
        self.return_object_features = return_object_features
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
            - if return_object_features: (B, N_objects, F)
            - else: (B, 512, 7, 7)
        """
        preprocessed_image = self.preprocess_image(image)
        features = self.cnn(preprocessed_image)  # shape: (B, 512, 7, 7)

        if self.return_object_features:
            B, C, H, W = features.size()
            features = features.view(B, C, H * W).permute(0, 2, 1)  # (B, 49, 512)
        return features

class FactoredRepresentationEncoder(nn.Module):
    """
    Pro-forma encoder for pre-processed factored state descriptions (object matrices).

    This module primarily ensures the input tensor has the correct format (dtype),
    acting as a standardized interface similar to BinaryQuestionEncoder. It does not
    perform computations like CNNImageEncoder.
    """
    def __init__(self):
        """
        This is a pro-forma module.
        """
        super(FactoredRepresentationEncoder, self).__init__()

    def forward(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processes the batch tensor of pre-computed object features.

        Parameters:
        -----------
        batch_tensor : torch.Tensor
            Input tensor containing batched object features.
            Expected shape: (batch_size, num_objects, feature_dim).
            Shape for specialized Sort-of-CLEVR-like case: (batch_size, 6, 12).

        Returns:
        --------
        torch.Tensor
            The input tensor, ensured to have dtype torch.float.
        """
        if not isinstance(batch_tensor, torch.Tensor):
            batch_tensor = torch.tensor(batch_tensor, dtype=torch.float)

        if batch_tensor.dtype != torch.float:
            batch_tensor = batch_tensor.float()

        return batch_tensor
        
class LSTMQuestionEncoder(nn.Module):
    """
    Module for encoding questions using an LSTM.


    Methods:
    -------
    forward(questions):
        Performs a forward pass through the network, returning the final hidden state of the LSTM.
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMQuestionEncoder, self).__init__()
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
        return hidden[-1]  # shape: (batch_size, hidden_size)

class BinaryQuestionEncoder(nn.Module):
    """
    Module for encoding binary questions as used in the Sort-of-CLEVR dataset.
    Converts binary question vectors to tensors for use with the RelationalNetwork.
    """
    def __init__(self, binary_length=11):
        """
        Initializes the BinaryQuestionEncoder.
        
        Parameters:
        -----------
        binary_length : int
            The length of the binary vector representing questions.
            For Sort-of-CLEVR, this is 11 (6 bits for color, 5 bits for question type).
        """
        super(BinaryQuestionEncoder, self).__init__()
        self.binary_length = binary_length
        
    def forward(self, binary_questions):
        """
        Processes binary question vectors.
        
        Parameters:
        ----------
        binary_questions : torch.Tensor
            Tensor of binary question vectors with shape (batch_size, binary_length).
            
        Returns:
        -------
        torch.Tensor
            The same tensor, ensuring it has the correct dtype and device.
        """

        if not isinstance(binary_questions, torch.Tensor):
            binary_questions = torch.tensor(binary_questions, dtype=torch.float)
        elif binary_questions.dtype != torch.float:
            binary_questions = binary_questions.float()
            
        return binary_questions 
        
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

    def __init__(self, question_dim, num_classes, img_feature_dim):
        """
        Initializes the RelationalNetwork with two MLPs: g_theta and f_phi.

        Parameters
        ----------
        question_dim : int
            The dimensionality of the question embedding.
        num_classes : int
            The number of output classes for the multi-class classification.        
        """
        super(RelationalNetwork, self).__init__()
        
        self.g_theta = nn.Sequential(
            nn.Linear(img_feature_dim + question_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU()
        )
        self.f_phi = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )
    
    def forward(self, object_features, question_embedding):
        """
        Vectorized forward pass, includes self-relations, aligning with paper.

        Parameters
        ----------
        object_features : torch.Tensor
            A tensor containing the features of objects.
            Tensor shape: (batch_size, num_objects, feature_dim).
        question_embedding : torch.Tensor
            A tensor representing the question embedding.
            Tensor shape: (batch_size, question_dim).

        Returns
        -------
        torch.Tensor
            Output tensor shape: (batch_size, num_classes).
        """

        batch_size, num_objects, feature_dim = object_features.size()

        # preparing object pair features

        # we want to create a batch of cubes, where the front-view square 
        # is formed by a set of objects going from the first row to the bottom one
        # and copy (broadcast) this single col num_objects times across all cols
        # and the depth of the cube represents the features
        obj_i = object_features.unsqueeze(2).repeat(1, 1, num_objects, 1)

        # here we apply the same idea, but now we start with objects spread across the first row and
        # we want to broadcast to all rows 
        obj_j = object_features.unsqueeze(1).repeat(1, num_objects, 1, 1)

        # now, let's prepare a full copy of the question embedding for each pair
        question_embedding_expanded = question_embedding.unsqueeze(1).unsqueeze(2).repeat(1, num_objects, num_objects, 1)

        # and finally to conclude this mental gymnastics, concat all the cubes - 
        # features of obj_i, obj_j and question embedding finally come together 
        # (batch_size, num_objects, num_objects, 2*feature_dim + question_dim)
        pair_features = torch.cat([obj_i, obj_j, question_embedding_expanded], dim=3)

        # and flatten pairs for processing: (batch_size * num_objects * num_objects, combined_feature_dim)
        pair_features = pair_features.view(batch_size * num_objects * num_objects, -1)

        # voila, compute g_theta in one go
        relations = self.g_theta(pair_features)

        # reshape and sum relations per image: (batch_size, num_objects * num_objects, hidden_dim)
        relations = relations.view(batch_size, num_objects * num_objects, -1)
        relations_sum = relations.sum(dim=1)

        # final processing via f_phi
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

    def __init__(self, img_arch, vocab_size, embed_size, hidden_size, num_layers, num_classes, question_form, image_form):
        """
        Initializes the RelationalReasoningModel with an ImageEncoder, QuestionEncoder, and RelationalNetwork.

        Parameters
        ----------
        img_arch: str
            The type of image encoder backbone - values: 'cnn' or 'resnet'. 
        vocab_size : int
            The size of the vocabulary used in the QuestionEncoder.
        embed_size : int
            The dimensionality of the word embeddings in the QuestionEncoder.
        hidden_size : int
            The number of features in the hidden state of the LSTM in the QuestionEncoder.
        num_layers : int
            The number of recurrent layers in the LSTM of the QuestionEncoder.
        num_classes : int
            The number of output classes for the multi-class classification.
        """
        super(RelationalReasoningModel, self).__init__()
        
        if image_form == 'image':
            if img_arch == 'cnn':
                self.image_encoder = CNNImageEncoder()
                self.img_feature_dim = 2 * (256 + 2)  # 1 object pair, each object has 256 features +2 for spatial coordinates (x, y)
            elif img_arch == 'resnet':
                self.image_encoder = ResNetImageEncoder(return_object_features=True)
                self.img_feature_dim = 2 * 512
            else:
                raise ValueError(f"Unsupported image architecture: {img_arch}")
        elif image_form == 'matrix':
            self.image_encoder = FactoredRepresentationEncoder()
            self.img_feature_dim = 2 * 12
        else:
            raise ValueError(f"Unsupported image form: {image_form}")
        
        if question_form == 'string':
            self.question_encoder = LSTMQuestionEncoder(vocab_size, embed_size, hidden_size, num_layers)
        elif question_form == 'binary':
            self.question_encoder = BinaryQuestionEncoder()
            hidden_size = 11
        else:
            raise ValueError(f"Unsupported question form: {question_form}")  

        self.relation_network = RelationalNetwork(hidden_size, num_classes, self.img_feature_dim)
    
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
        object_features = self.image_encoder(image)
        # flatten the spatial dimensions to produce a set of object features
        # object_features = image_features.view(image_features.size(0), -1, image_features.size(1))
        question_embedding = self.question_encoder(question)
        # perform relational reasoning using the object features and question embedding
        output = self.relation_network(object_features, question_embedding)
        
        return output

class BaselineModel(nn.Module):
    """
    Baseline model that combines a CNN-based image encoder and a question encoder
    (either LSTM-based or binary) with a standard MLP for classification — without
    any relational reasoning module. Matches the CNN+MLP baseline from the Sort-of-CLEVR paper.
    """

    def __init__(self, img_arch, vocab_size, embed_size, hidden_size, num_layers, num_classes, question_form, image_form):
        super(BaselineModel, self).__init__()

        # image encoder
        if image_form == 'image':
            if img_arch == 'cnn':
                self.image_encoder = CNNImageEncoder()  # output shape: (batch_size, 256, 5, 5)
                self.img_feature_dim = 5 * 5 * (256 + 2)  # +2 for spatial coordinates (x, y)
            elif img_arch == 'resnet':
                self.image_encoder = ResNetImageEncoder(return_object_features=False)  # output shape: (batch_size, 512, 7, 7)
                self.img_feature_dim = 512 * 7 * 7
            else:
                raise ValueError(f"Unsupported image architecture: {img_arch}")
        elif image_form == 'matrix':
            self.image_encoder = FactoredRepresentationEncoder() # output shape: (batch_size, 6, 12)
            self.img_feature_dim = 6 * 12
        else:
            raise ValueError(f"Unsupported image form: {image_form}")

        # question encoder
        if question_form == 'string':
            self.question_encoder = LSTMQuestionEncoder(vocab_size, embed_size, hidden_size, num_layers)
        elif question_form == 'binary':
            self.question_encoder = BinaryQuestionEncoder()
            hidden_size = 11  # fixed for binary question format
        else:
            raise ValueError(f"Unsupported question form: {question_form}")

        # MLP for classification (aligned with paper: 4 layers, 2000 neurons each, ReLU)
        input_dim = self.img_feature_dim + hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, num_classes)  # final logits for classification
        )

    def forward(self, image: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get the model's prediction.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (batch_size, C, H, W).
        questions : torch.Tensor
            Input question tensor (either sequence of word indices or binary vector).

        Returns
        -------
        torch.Tensor
            Output prediction tensor of shape (batch_size, num_classes).
        """

        # encode image and flatten
        image_features = self.image_encoder(image)  # shape: (B, C, H, W)
        image_features = image_features.reshape(image_features.size(0), -1)  # shape: (B, img_feature_dim)

        # encode question
        question_embedding = self.question_encoder(questions)

        # combine and classify
        combined = torch.cat((image_features, question_embedding), dim=1)  # shape: (B, img_feature_dim + hidden_size)
        output = self.classifier(combined)
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
                img_arch=kwargs['img_arch'],
                vocab_size=kwargs['vocab_size'],
                embed_size=kwargs['embed_size'],
                hidden_size=kwargs['hidden_size'],
                num_layers=kwargs['num_layers'],
                num_classes=kwargs['num_classes'],
                question_form=kwargs['question_form'],
                image_form=kwargs['image_form']
                
            )
        elif model_type == 'relational':
            return RelationalReasoningModel(
                img_arch=kwargs['img_arch'],
                vocab_size=kwargs['vocab_size'],
                embed_size=kwargs['embed_size'],
                hidden_size=kwargs['hidden_size'],
                num_layers=kwargs['num_layers'],
                num_classes=kwargs['num_classes'],
                question_form=kwargs['question_form'],
                image_form=kwargs['image_form']
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