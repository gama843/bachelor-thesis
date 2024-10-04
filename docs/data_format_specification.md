## Description

The `descr.json` structure contains ground-truth labels for a specific dataset instance. It includes a list of objects present in each image, each with specific attributes like color, bounding box location, object type, and distances to other objects. Additionally, it includes a series of questions related to the objects and their corresponding answers. It also contains a factored image representation, the image description matrix, which provides a way to train a model without a visual module. Similarly, the questions are represented as binary vectors, in addition to the natural language form, in case you would like to train a model without an NLP module.

### JSON Structure

#### Root
- **Key**: (string) The relative file path of the image (starting from the generation script file).
- **Value**: (array) A list of objects and questions related to the image.

#### Objects

- **id**: (integer) A unique identifier for the object.
- **color**: (array of 3 integers) The RGB color values of the object.
    - Example: `[255, 0, 0]` for red.
- **bboxlocation**: (array of 2 arrays of 2 integers) The bounding box coordinates of the object.
    - Each sub-array represents the top-left and bottom-right corners of the bounding box where (0,0) is taken as the top-left corner of the image.
    - Example: `[[67, 35], [82, 93]]`.
- **object_type**: (string) The type of the object (e.g., "triangle", "square", "circle").
- **color_name**: (string) The name of the color of the object.
- **distances**: (array) A list of Euclidean distances to other objects, calculated as the distance between the centers of the objects' bounding boxes.
    - Each distance is represented by:
        - **object_id**: (integer) The ID of the other object.
        - **object_type**: (string) The type of the other object.
        - **distance**: (float) The Euclidean distance to the other object.

#### Questions and Answers

- **question**: (string) The question related to the objects.
- **question_vector**: (array of 11 integers) A vector representation of the question. The first 6 bits correspond to a one-hot vector for the color of the object, another 2 bits are a one-hot vector for the question category, and the last 3 bits identify the subtype of the question in the given category.
- **answer**: (string/integer) The answer to the question.

**Question templates:**

**a) relational questions**
1. What is the color of the object that is closest to the COLOR object?
2. What is the shape that is farthest from the COLOR object?
3. How many objects have the shape of the COLOR object?

**b) non-relational questions**
1. Is the COLOR object on the top or on the bottom?
2. Is the COLOR object on the left or on the right?
3. What is the shape of the COLOR object?


#### Image Description

- **image_description_matrix**: (optional, array) A matrix describing the objects in the image, used for training a model without a visual module.
    - Each entry is an array containing:
        - **color_name**: (string) The name of the color of the object.
        - **object_type**: (string) The type of the object.
        - **bboxlocation (flattened)**: (4 integers) The bounding box coordinates of the object, flattened.
            - Example: `[67, 35, 82, 93]`.

### Example

```json
{
    "sample_data/1.png": [
        {
            "id": 1,
            "color": [255, 0, 0],
            "bboxlocation": [[67, 35], [82, 93]],
            "object_type": "triangle",
            "color_name": "blue",
            "distances": [
                {"object_id": 2, "object_type": "square", "distance": 36.88},
                {"object_id": 3, "object_type": "triangle", "distance": 49.74},
                {"object_id": 4, "object_type": "square", "distance": 61.29},
                {"object_id": 5, "object_type": "circle", "distance": 32.8},
                {"object_id": 6, "object_type": "circle", "distance": 65.86}
            ]
        },
        {
            "id": 2,
            "color": [0, 0, 255],
            "bboxlocation": [[37, 31], [56, 50]],
            "object_type": "square",
            "color_name": "red",
            "distances": [
                {"object_id": 1, "object_type": "triangle", "distance": 36.88},
                {"object_id": 3, "object_type": "triangle", "distance": 51.24},
                {"object_id": 4, "object_type": "square", "distance": 26.48},
                {"object_id": 5, "object_type": "circle", "distance": 54.15},
                {"object_id": 6, "object_type": "circle", "distance": 33.38}
            ]
        },
        {
            "question": "What is the shape of the object that is farthest from the blue object?",
            "question_vector": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            "answer": "circle"
        },
        {
            "question": "Is the blue object on the top or the bottom?",
            "question_vector": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            "answer": "bottom",
            "image_description_matrix": [
                ["blue", "triangle", 67, 35, 82, 93],
                ["red", "square", 37, 31, 56, 50],
                ["green", "triangle", 16, 76, 46, 102],
                ["orange", "square", 10, 25, 30, 45],
                ["pink", "circle", 93, 37, 107, 51],
                ["yellow", "circle", 34, 0, 48, 14]
            ]
        }
    ]
}