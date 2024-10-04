import os
import math
import json
import shutil
from time import time
import numpy as np
import cv2
import progressbar

# This module uses the @public annotation to include certain private methods in the generated documentation.
# The @public tag is applied only for the purpose of documentation generation.

class DataGenerator:
    """
    A class to generate synthetic datasets for relational reasoning tasks.

    Attributes:
    -----------
    base_path : str
        The base directory to save generated images and JSON files.
    img_dim : int
        The dimension (width and height) of the generated images.
    num_images : int
        The number of images to generate.
    min_objects : int
        Minimum number of objects to include in each image.
    max_objects : int
        Maximum number of objects to include in each image.
    visualize_bboxes : bool
        Whether to visualize bounding boxes of the objects.
    add_noise : bool
        Whether to add noise to the images.
    noise_prob : float
        Probability of noise presence in the images.
    fixed_colors : bool
        Whether to use a fixed set of colors from the palette.
    fill_objects : bool
        Whether the objects should be filled or not.
    shapes : dict
        A dictionary of shape generation functions.
    palette : dict
        A color palette for objects.
    reusable_palette_colors : bool
        Whether colors from the palette can be reused.
    """

    def __init__(self, 
                 base_path='./data', 
                 img_dim=128, 
                 num_images=100, 
                 min_objects=0, 
                 max_objects=6, 
                 visualize_bboxes=True, 
                 add_noise=False, 
                 noise_prob=0.05, 
                 fixed_colors=True, 
                 fill_objects=False, 
                 reusable_palette_colors=False, 
                 palette=None):
        self.base_path = base_path
        self.img_dim = img_dim
        self.num_images = num_images
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.visualize_bboxes = visualize_bboxes
        self.add_noise = add_noise
        self.noise_prob = noise_prob
        self.fixed_colors = fixed_colors
        self.fill_objects = fill_objects
        self.reusable_palette_colors = reusable_palette_colors
        self.palette = palette or {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'orange': (0, 165, 255),
            'pink': (243, 194, 246),
            'yellow': (0, 255, 255)
        }
        self.shapes = {
            "rectangle": self._gen_rectangle,
            "circle": self._gen_circle,
            "triangle": self._gen_triangle,
            "pentagon": self._gen_pentagon,
            "square": self._gen_square
        }
        self._prepare_directory()

    def _prepare_directory(self):
        """
        @public
        
        Prepare the base directory for saving images and JSON files.
        """
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
        os.makedirs(self.base_path)

    def _gen_rectangle(self, img, color, fill):
        """
        @public

        Generate a rectangle on the given image with the specified color and fill.

        Parameters:
        -----------
        img : numpy.ndarray
            The input image on which to draw the rectangle.
        color : tuple
            A tuple representing the color of the rectangle in BGR format (e.g., (255, 0, 0) for blue).
        fill : bool
            A boolean indicating whether the rectangle should be filled (True) or outlined (False).

        Returns:
        --------
        img : numpy.ndarray
            The image with the rectangle drawn on it.
        bbox : tuple
            A tuple containing two points that represent the bounding box of the rectangle
            ((x1, y1), (x2, y2)).
        """
        thickness = -1 if fill else np.random.randint(1, 3)
        x1 = np.random.randint(0 + thickness, self.img_dim / 2 - thickness)
        y1 = np.random.randint(0 + thickness, self.img_dim / 2 - thickness)
        x2 = np.random.randint(x1 + 10, self.img_dim - thickness - 10)
        y2 = np.random.randint(y1 + 10, self.img_dim - thickness - 10)
        
        #making sure it's not a square
        while y2 == y1 + np.random.choice([-1, 1]) * abs(x1 - x2):
            y2 = np.random.randint(y1 + 10, self.img_dim - thickness - 10)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        return img, ((x1, y1), (x2, y2))        
    
    def _gen_square(self, img, color, fill):
        """
        @public

        Generate a square on the given image with the specified color and fill.

        Parameters:
        -----------
        img : numpy.ndarray
            The input image on which to draw the square.
        color : tuple
            A tuple representing the color of the square in BGR format (e.g., (255, 0, 0) for blue).
        fill : bool
            A boolean indicating whether the square should be filled (True) or outlined (False).

        Returns:
        --------
        img : numpy.ndarray
            The image with the square drawn on it.
        bbox : tuple
            A tuple containing two points that represent the bounding box of the square
            ((x1, y1), (x2, y2)).
        """
        thickness = -1 if fill else np.random.randint(1, 3)

        x1 = np.random.randint(0 + thickness, self.img_dim / 2 - thickness)
        y1 = np.random.randint(0 + thickness, self.img_dim / 2 - thickness)

        side_length = np.random.randint(10, self.img_dim / 2 - thickness)

        x2 = x1 + side_length
        y2 = y1 + side_length

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        return img, ((x1, y1), (x2, y2))
    
    def _gen_circle(self, img, color, fill):
        """
        @public

        Generate a circle on the given image with the specified color and fill.

        Parameters:
        -----------
        img : numpy.ndarray
            The input image on which to draw the circle.
        color : tuple
            A tuple representing the color of the circle in BGR format (e.g., (255, 0, 0) for blue).
        fill : bool
            A boolean indicating whether the circle should be filled (True) or outlined (False).

        Returns:
        --------
        img : numpy.ndarray
            The image with the circle drawn on it.
        bbox : tuple
            A tuple containing two points that represent the bounding box of the circle
            ((bb_x1, bb_y1), (bb_x2, bb_y2)).
        """
        radius = np.random.randint(5, self.img_dim // 7)
        thickness = -1 if fill else np.random.randint(1, 3)

        center_coordinates = (np.random.randint(1 + radius + thickness, self.img_dim - radius - thickness - 1),
                            np.random.randint(1 + radius + thickness, self.img_dim - radius - thickness - 1))
        cv2.circle(img, center_coordinates, radius, color, thickness)
        bb_x1 = center_coordinates[0] - radius
        bb_y1 = center_coordinates[1] - radius
        bb_x2 = center_coordinates[0] + radius
        bb_y2 = center_coordinates[1] + radius
        
        return img, ((bb_x1, bb_y1), (bb_x2, bb_y2))    
    
    def _gen_triangle(self, img, color, fill):
        """
        @public

        Generate a triangle on the given image with the specified color and fill.

        Parameters:
        -----------
        img : numpy.ndarray
            The input image on which to draw the triangle.
        color : tuple
            A tuple representing the color of the triangle in BGR format (e.g., (255, 0, 0) for blue).
        fill : bool
            A boolean indicating whether the triangle should be filled (True) or outlined (False).

        Returns:
        --------
        img : numpy.ndarray
            The image with the triangle drawn on it.
        bbox : tuple
            A tuple containing two points that represent the bounding box of the triangle
            ((min_x, min_y), (max_x, max_y)).
        """
        thickness = np.random.randint(1, 3)
        boundary_offset = 10

        # adjust the range for points to ensure they are within the boundary offset
        line1_start_point = (
            np.random.randint(boundary_offset + thickness, self.img_dim - boundary_offset - thickness - 1),
            np.random.randint(boundary_offset + thickness, self.img_dim - boundary_offset - thickness - 1)
        )
        line1_end_point = (
            np.random.randint(boundary_offset + thickness, self.img_dim - boundary_offset - thickness - 1),
            line1_start_point[1]
        )

        # ensure the base is at least 10px long
        while abs(line1_end_point[0] - line1_start_point[0]) < 10:
            line1_end_point = (
                np.random.randint(boundary_offset + thickness, self.img_dim - boundary_offset - thickness - 1),
                line1_start_point[1]
            )

        # ensure the apex is at least 10px away from the base and within boundaries
        apex_y = line1_start_point[1] + np.random.choice([-1, 1]) * np.random.randint(10, self.img_dim - 2 * boundary_offset - thickness - 1)
        apex_x = np.random.randint(
            min(line1_start_point[0], line1_end_point[0]),
            max(line1_start_point[0], line1_end_point[0])
        )
        
        # adjust apex_y if it's outside the allowed range
        while apex_y < boundary_offset + thickness or apex_y > self.img_dim - boundary_offset - thickness - 1:
            apex_y = line1_start_point[1] + np.random.choice([-1, 1]) * np.random.randint(10, self.img_dim - 2 * boundary_offset - thickness - 1)
            apex_x = np.random.randint(
                min(line1_start_point[0], line1_end_point[0]),
                max(line1_start_point[0], line1_end_point[0])
            )
            
        apex_point = (apex_x, apex_y)

        # draw the triangle lines
        cv2.line(img, line1_start_point, line1_end_point, color, thickness)
        cv2.line(img, line1_start_point, apex_point, color, thickness)
        cv2.line(img, line1_end_point, apex_point, color, thickness)
        
        # fill the triangle if required
        if fill:
            triangle_cnt = np.array([line1_start_point, line1_end_point, apex_point])
            cv2.fillPoly(img, [triangle_cnt], color)

        # calculate the bounding box of the triangle
        min_x = min(line1_start_point[0], line1_end_point[0], apex_point[0])
        max_x = max(line1_start_point[0], line1_end_point[0], apex_point[0])
        min_y = min(line1_start_point[1], line1_end_point[1], apex_point[1])
        max_y = max(line1_start_point[1], line1_end_point[1], apex_point[1])

        return img, ((min_x, min_y), (max_x, max_y))    
    
    def _gen_pentagon(self, img, color, fill):
        """
        @public

        Generate a pentagon on the given image with the specified color and fill.

        Parameters:
        -----------
        img : numpy.ndarray
            The input image on which to draw the pentagon.
        color : tuple
            A tuple representing the color of the pentagon in BGR format (e.g., (255, 0, 0) for blue).
        fill : bool
            A boolean indicating whether the pentagon should be filled (True) or outlined (False).

        Returns:
        --------
        img : numpy.ndarray
            The image with the pentagon drawn on it.
        bbox : tuple
            A tuple containing two points that represent the bounding box of the pentagon
            ((min_x, min_y), (max_x, max_y)).
        """
        thickness = np.random.randint(1, 3)
        radius = self.img_dim // 7
        num_sides = 5
        rotation = np.random.uniform(0, 2 * np.pi)
        center_x = np.random.randint(radius, self.img_dim - radius)
        center_y = np.random.randint(radius, self.img_dim - radius)

        # calculate the coordinates of the pentagon vertices
        points = []
        for j in range(num_sides):
            angle = 2 * np.pi * j / num_sides + rotation
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            points.append((x, y))

        while not self._is_inside_image(points, self.img_dim):
            center_x = np.random.randint(radius, self.img_dim - radius)
            center_y = np.random.randint(radius, self.img_dim - radius)

            points = []
            for k in range(num_sides):
                angle = 2 * np.pi * k / num_sides + rotation
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                points.append((x, y))

        points_array = np.array(points, dtype=np.int32)
        cv2.polylines(img, [points_array], isClosed=True, color=color, thickness=thickness)
        
        if fill:
            pentagon_cnt = np.array(points_array)
            cv2.fillPoly(img, [pentagon_cnt], color)
            
        min_x = min(points, key=lambda t: t[0])[0]
        max_x = max(points, key=lambda t: t[0])[0]
        min_y = min(points, key=lambda t: t[1])[1]
        max_y = max(points, key=lambda t: t[1])[1]
        
        return img, ((min_x, min_y), (max_x, max_y))
    
    def _generate_img(self, min_objects, max_objects, visualize_bboxes, add_noise, noise_prob, fixed_colors, fill_objects, shapes, palette, reusable_palette_colors):
        """
        @public

        Generate an image with randomly placed shapes according to specified parameters.

        Parameters:
        -----------
        min_objects : int
            Minimum number of objects to include in the image.
        max_objects : int
            Maximum number of objects to include in the image.
        visualize_bboxes : bool
            Whether to draw bounding boxes around the shapes.
        add_noise : bool
            Whether to add noise to the image.
        noise_prob : float
            The probability of noise presence in the image.
        fixed_colors : bool
            Whether to use a fixed set of colors from the palette.
        fill_objects : bool
            Whether the shapes should be filled (True) or outlined (False).
        shapes : dict
            A dictionary of shape generation functions.
        palette : dict
            A color palette for the shapes.
        reusable_palette_colors : bool
            Whether colors from the palette can be reused.

        Returns:
        --------
        img : numpy.ndarray
            The generated image with the specified number of shapes.
        objects_info : list
            A list of dictionaries containing information about each shape in the image.
        """
        img = np.zeros((self.img_dim, self.img_dim, 3))
        existing_boxes = []
        attempts = 0
        max_attempts = max_objects * 10
        added_objects = 0
        objects_info = []
        object_id = 1
        colors = list(palette.keys())
        
        if add_noise:
            img = self._gen_noise(img, noise_prob)
        
        if min_objects == max_objects:
            num_objects = min_objects
        else:
            num_objects = np.random.randint(min_objects, max_objects)
        
        while added_objects < num_objects:
            # choose the color of the object
            if fixed_colors:
                random_color_key = np.random.choice(colors)
                if not reusable_palette_colors:
                    colors.remove(random_color_key)
                color = palette[random_color_key]
            else:
                color = (np.random.randint(1, 255), np.random.randint(1, 255), np.random.randint(1, 255))
            
            # choose object type
            shape = np.random.choice(list(shapes.keys()))
            shape_func = shapes[shape]
            
            img_backup = img.copy()
            successful_draw = False
            
            # generate a shape and get its bounding box using a copy of the current image
            tmp_img = img.copy()
            new_img, bounding_box = shape_func(tmp_img, color, fill=fill_objects)

            if not any(self._do_overlap(bounding_box, box) for box in existing_boxes):
                img = new_img
                existing_boxes.append(bounding_box)

                objects_info.append({
                    "id": object_id,
                    "color": color,
                    "bboxlocation": bounding_box,
                    "object_type": shape
                })
                
                if fixed_colors:
                    objects_info[-1].update({"color_name": random_color_key})

                object_id += 1

                if visualize_bboxes:
                    self._draw_bounding_box(img, bounding_box)  

                successful_draw = True
                added_objects += 1
            
            if not successful_draw:
                img = img_backup
                colors.append(random_color_key)
                
            attempts += 1
            
            if attempts == max_attempts:
                img, objects_info = self._generate_img(min_objects, max_objects, visualize_bboxes, add_noise, noise_prob, fixed_colors, fill_objects, shapes, palette, reusable_palette_colors)
                break    
        
        objects_info = self._add_pairwise_distances(objects_info)
        
        return img, objects_info
    
    def _draw_bounding_box(self, img, bounding_box, color=(255, 255, 255), thickness=1):
        """
        @public

        Draw a bounding box on the given image.

        Parameters:
        -----------
        img : numpy.ndarray
            The input image on which to draw the bounding box.
        bounding_box : tuple
            A tuple containing two points that represent the bounding box
            ((x1, y1), (x2, y2)).
        color : tuple, optional
            A tuple representing the color of the bounding box in BGR format (default is white).
        thickness : int, optional
            The thickness of the bounding box lines (default is 1).

        Returns:
        --------
        img : numpy.ndarray
            The image with the bounding box drawn on it.
        """
        start_point, end_point = bounding_box
        cv2.rectangle(img, start_point, end_point, color, thickness)
        
        return img

    def _euclidean_distance(self, point1, point2):
        """
        @public

        Calculate the Euclidean distance between two points.

        Parameters:
        -----------
        point1 : tuple
            A tuple representing the coordinates of the first point (x1, y1).
        point2 : tuple
            A tuple representing the coordinates of the second point (x2, y2).

        Returns:
        --------
        float
            The Euclidean distance between the two points, rounded to two decimal places.
        """
        return round(math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2), 2)

    def _compute_bbox_center(self, bbox):
        """
        @public

        Compute the center point of a bounding box.

        Parameters:
        -----------
        bbox : tuple
            A tuple containing two points that represent the bounding box ((x1, y1), (x2, y2)).

        Returns:
        --------
        tuple
            A tuple representing the coordinates of the center point of the bounding box (center_x, center_y).
        """
        return ((bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2)

    def _gen_noise(self, img, prob):
        """
        @public

        Generate random noise on the given image based on a specified probability.

        Parameters:
        -----------
        img : numpy.ndarray
            The input image on which to add noise.
        prob : float
            The probability of each pixel being replaced by noise (value between 0 and 1).

        Returns:
        --------
        output : numpy.ndarray
            The image with random noise added to it.
        """
        noise_mask = np.random.choice([True, False], size=img.shape, p=[prob, 1-prob])
        noise = np.random.randint(1, 256, size=img.shape, dtype=np.uint8)
        output = np.where(noise_mask, noise, img)
        
        return output

    def _find_line_endpoints(self, center_x, center_y, slope, length):
        """
        @public

        Find the endpoints of a line segment given its center, slope, and length.

        Parameters:
        -----------
        center_x : float
            The x-coordinate of the center point of the line.
        center_y : float
            The y-coordinate of the center point of the line.
        slope : float
            The slope of the line.
        length : float
            The total length of the line segment.

        Returns:
        --------
        tuple
            A tuple containing the start point (start_x, start_y) and end point (end_x, end_y) of the line.
        """
        # calculate the angle of the line with respect to the x-axis
        angle = math.atan(slope)

        # calculate the distance from the center point to the start and end points
        dx = 0.5 * length * math.cos(angle)
        dy = 0.5 * length * math.sin(angle)

        # calculate the start and end point coordinates
        start_x = int(center_x - dx)
        start_y = int(center_y - dy)
        end_x = int(center_x + dx)
        end_y = int(center_y + dy)

        return (start_x, start_y), (end_x, end_y)
    
    def _get_bounding_rect(self, points):
        """
        @public

        Compute the bounding rectangle for a given set of points.

        Parameters:
        -----------
        points : list of tuples
            A list of tuples, where each tuple represents the (x, y) coordinates of a point.

        Returns:
        --------
        tuple
            A tuple containing two points that represent the top-left and bottom-right corners
            of the bounding rectangle ((min_x, min_y), (max_x, max_y)).
        """
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        
        return (min(x_coords), min(y_coords)), (max(x_coords), max(y_coords))
    
    def _is_inside_image(self, points, img_dim):
        """
        @public

        Check if a set of points is fully inside the boundaries of an image.

        Parameters:
        -----------
        points : list of tuples
            A list of tuples, where each tuple represents the (x, y) coordinates of a point.
        img_dim : int
            The dimension (width and height) of the image.

        Returns:
        --------
        bool
            True if all points are inside the image boundaries, False otherwise.
        """
        bounding_rect = self._get_bounding_rect(points)
        
        return bounding_rect[0][0] >= 0 and bounding_rect[0][1] >= 0 and bounding_rect[1][0] <= img_dim and bounding_rect[1][1] <= img_dim
    
    def _do_overlap(self, box1, box2):
        """
        @public

        Check if two bounding boxes overlap with an offset applied to all sides.

        Parameters:
        -----------
        box1 : tuple
            A tuple containing two points that represent the first bounding box ((x1, y1), (x2, y2)).
        box2 : tuple
            A tuple containing two points that represent the second bounding box ((x1, y1), (x2, y2)).

        Returns:
        --------
        bool
            True if the bounding boxes overlap, False otherwise.
        """
        # check if two bounding boxes overlap with an offset applied to all sides
        offset = 3

        expanded_box1 = ((box1[0][0] - offset, box1[0][1] - offset), (box1[1][0] + offset, box1[1][1] + offset))
        expanded_box2 = ((box2[0][0] - offset, box2[0][1] - offset), (box2[1][0] + offset, box2[1][1] + offset))

        x_overlaps = (expanded_box1[0][0] < expanded_box2[1][0]) and (expanded_box1[1][0] > expanded_box2[0][0])
        y_overlaps = (expanded_box1[0][1] < expanded_box2[1][1]) and (expanded_box1[1][1] > expanded_box2[0][1])
        
        return x_overlaps and y_overlaps
    
    def _add_pairwise_distances(self, objects_info):
        """
        @public

        Add pairwise distances between all objects in the image to their information.

        Parameters:
        -----------
        objects_info : list
            A list of dictionaries, where each dictionary contains information about an object 
            in the image, including its bounding box location.

        Returns:
        --------
        list
            The updated list of dictionaries with pairwise distances added for each object.
        """
        for obj_info in objects_info:
            # skip the question object
            if 'question' in obj_info:
                continue
            bbox_center = self._compute_bbox_center(obj_info["bboxlocation"])
            distances = []
            for other_obj_info in objects_info:
                # skip the question object
                if 'question' in other_obj_info:
                    continue
                if other_obj_info["id"] != obj_info["id"]:  # don't compare with itself
                    other_bbox_center = self._compute_bbox_center(other_obj_info["bboxlocation"])
                    distance = self._euclidean_distance(bbox_center, other_bbox_center)
                    distances.append({
                        "object_id": other_obj_info["id"],
                        "object_type": other_obj_info["object_type"],
                        "distance": distance
                    })
            obj_info["distances"] = distances
        
        return objects_info
    
    def _add_questions_and_answers(self, img_dim, objects_info, palette, shapes):
        """
        @public

        Add relational and non-relational questions and answers to the image data.

        This method generates 10 relational and 10 non-relational questions based on the objects
        in the image and appends these questions, along with their vectors and answers, to the objects' information.

        Parameters:
        -----------
        img_dim : int
            The dimension (width and height) of the image.
        objects_info : list
            A list of dictionaries, where each dictionary contains information about an object 
            in the image, including its bounding box location.
        palette : dict
            A dictionary representing the color palette used for the objects.
        shapes : dict
            A dictionary of shape generation functions.

        Returns:
        --------
        list
            The updated list of dictionaries with questions and answers added for each image.
        """
        # add 10 relational and 10 non-relational questions per image
        vectors = set()
        while len(vectors) < 10:
            
            question_vector = self._get_random_question_vector(relational=True)
            
            # create a string representation of the vector
            vector_string = ''.join([str(bit) for bit in question_vector])
            if vector_string in vectors:
                continue
            vectors.add(vector_string)

            subtype = self._get_question_subtype(question_vector, relational=True)
            color = self._get_color_from_vector(question_vector, palette)
            question_text = self._get_question_text(subtype, color, relational=True)
            answer = self._get_relational_answer(subtype, color, objects_info)
            objects_info.append({
                "question": question_text,
                "question_vector": question_vector,
                "answer": answer
            })

        vectors = set()
        while len(vectors) < 10:
            
            question_vector = self._get_random_question_vector(relational=False)
            
            # create a string representation of the vector
            vector_string = ''.join([str(bit) for bit in question_vector])
            if vector_string in vectors:
                continue
            vectors.add(vector_string)

            subtype = self._get_question_subtype(question_vector, relational=False)
            color = self._get_color_from_vector(question_vector, palette)
            question_text = self._get_question_text(subtype, color, relational=False)
            answer = self._get_non_relational_answer(img_dim, subtype, color, objects_info)
            objects_info.append({
                "question": question_text,
                "question_vector": question_vector,
                "answer": answer
            })

        return objects_info
    
    def _get_random_question_vector(self, relational=True):
        """
        @public

        Generate a random question vector for relational or non-relational questions.

        As per the paper, the question vector is an 11-dimensional one-hot vector composed of:
        - A 6-bit one-hot vector for the color.
        - A 2-bit one-hot vector for the question subtype.
        - A 3-bit one-hot vector for the specific question.

        Parameters:
        -----------
        relational : bool, optional
            Whether the question vector is for a relational question (True) or non-relational question (False).

        Returns:
        --------
        list
            An 11-dimensional list representing the question vector.
        """
        # as per the paper, the question vector is a 11-dimensional one-hot vector, composed of 6-bit one-hot vector for the color, 
        # 2-bit one-hot vector of the question subtype and 3-bit one-hot for the specific question
        vector = [0] * 11

        six_bit_index = np.random.randint(0, 6)
        three_bit_index = np.random.randint(0, 3) + 8

        vector[six_bit_index] = 1
        if relational:
            vector[6] = 1
        else:
            vector[7] = 1
        vector[three_bit_index] = 1

        return vector
    
    def _get_question_subtype(self, vector, relational=True):
        """
        @public

        Decide the subtype of a question based on the given question vector.

        Parameters:
        -----------
        vector : list
            An 11-dimensional list representing the question vector.
        relational : bool, optional
            Whether the question is relational (True) or non-relational (False).

        Returns:
        --------
        str
            The subtype of the question. For relational questions, the subtype can be "closest", 
            "furthest", or "count". For non-relational questions, the subtype can be "topbottom", 
            "leftright", or "shape".

        Raises:
        -------
        ValueError
            If the vector does not correspond to any known question subtype.
        """
        if relational:
            if vector[8] == 1:
                return "closest"
            elif vector[9] == 1:
                return "furthest"
            elif vector[10] == 1:
                return "count"
            else:
                raise ValueError("Vector does not correspond to any known question subtype")
        else:
            if vector[8] == 1:
                return "topbottom"
            elif vector[9] == 1:
                return "leftright"
            elif vector[10] == 1:
                return "shape"
            else:
                raise ValueError("Vector does not correspond to any known question subtype")

    def _get_question_text(self, subtype, color, relational=True):
        """
        @public

        Generate the text of a question based on its subtype and color.

        Parameters:
        -----------
        subtype : str
            The subtype of the question (e.g., "closest", "furthest", "count" for relational questions, 
            or "topbottom", "leftright", "shape" for non-relational questions).
        color : str
            The color of the object that is the subject of the question.
        relational : bool, optional
            Whether the question is relational (True) or non-relational (False).

        Returns:
        --------
        str
            The text of the generated question.
        """
        if relational:
            if subtype == "closest":
                return f"What is the color of the object that is closest to the {color} object?"
            elif subtype == "furthest":
                return f"What is the shape of the object that is farthest from the {color} object?"
            elif subtype == "count":
                return f"How many objects have the shape of the {color} object?"
            else:
                return "Unknown question subtype"
        else:
            if subtype == "topbottom":
                return f"Is the {color} object on the top or the bottom?"
            elif subtype == "leftright":
                return f"Is the {color} object on the left or the right?"
            elif subtype == "shape":
                return f"What is the shape of the {color} object?"
            else:
                return "Unknown question subtype"
            
    def _get_color_from_vector(self, vector, palette):
        """
        @public

        Determine the color from the question vector using the given palette.

        Parameters:
        -----------
        vector : list
            An 11-dimensional list representing the question vector.
        palette : dict
            A dictionary representing the color palette, where keys are color names.

        Returns:
        --------
        str
            The name of the color corresponding to the one-hot encoded vector.

        Raises:
        -------
        ValueError
            If the vector does not correspond to any known color in the palette.
        """
        try:
            for i in range(0, 6):
                if vector[i] == 1:
                    return list(palette.keys())[i]
        except:
            raise ValueError("Unknown color")
        
    def _get_closest_object(self, color, objects_info):
        """
        @public

        Find the object that is closest to the specified color in the list of objects.

        Parameters:
        -----------
        color : str
            The color of the reference object to find the closest object to.
        objects_info : list
            A list of dictionaries, where each dictionary contains information about an object 
            in the image, including its color and distances to other objects.

        Returns:
        --------
        dict or None
            A dictionary containing information about the closest object to the specified color, 
            or None if no such object is found.
        """
        min_distance = float("inf")
        closest_object = None
        for obj_info in objects_info:
            # skip the question object
            if 'question' in obj_info:
                continue
            if obj_info["color_name"] == color:
                for distance_info in obj_info["distances"]:
                    if distance_info["distance"] < min_distance:
                        min_distance = distance_info["distance"]
                        closest_object_id = distance_info["object_id"]
        for obj_info in objects_info:
            if obj_info["id"] == closest_object_id:
                closest_object = obj_info
                break
        return closest_object
    
    def _get_furthest_object(self, color, objects_info):
        """
        @public

        Find the object that is furthest from the specified color in the list of objects.

        Parameters:
        -----------
        color : str
            The color of the reference object to find the furthest object from.
        objects_info : list
            A list of dictionaries, where each dictionary contains information about an object 
            in the image, including its color and distances to other objects.

        Returns:
        --------
        dict or None
            A dictionary containing information about the furthest object from the specified color, 
            or None if no such object is found.
        """
        max_distance = 0
        furthest_object = None
        for obj_info in objects_info:
            # skip the question object
            if 'question' in obj_info:
                continue
            if obj_info["color_name"] == color:
                for distance_info in obj_info["distances"]:
                    if distance_info["distance"] > max_distance:
                        max_distance = distance_info["distance"]
                        furthest_object_id = distance_info["object_id"]
        for obj_info in objects_info:
            if obj_info["id"] == furthest_object_id:
                furthest_object = obj_info
                break
        return furthest_object
    
    def _get_count_of_objects_with_shape(self, color, objects_info):
        """
        @public

        Count the number of objects with the same shape as the object of the specified color.

        Parameters:
        -----------
        color : str
            The color of the reference object to find the shape of.
        objects_info : list
            A list of dictionaries, where each dictionary contains information about an object 
            in the image, including its color and shape.

        Returns:
        --------
        int
            The count of objects that have the same shape as the object of the specified color.
        """
        # find the shape of the object in question
        shape = None
        for obj_info in objects_info:
            # skip the question object
            if 'question' in obj_info:
                continue
            if obj_info["color_name"] == color:
                shape = obj_info["object_type"]
                break
        # count the number of objects with the same shape
        count = 0
        for obj_info in objects_info:
            # skip the question object
            if 'question' in obj_info:
                continue
            if obj_info["object_type"] == shape:
                count += 1
        return count
    
    def _get_relational_answer(self, subtype, color, objects_info):
        """
        @public

        Determine the answer to a relational question based on its subtype and reference color.

        Parameters:
        -----------
        subtype : str
            The subtype of the question (e.g., "closest", "furthest", "count").
        color : str
            The color of the reference object for the question.
        objects_info : list
            A list of dictionaries, where each dictionary contains information about an object 
            in the image, including its color, shape, and distances to other objects.

        Returns:
        --------
        str or int
            The answer to the relational question. This could be a color name, object type, or a count 
            depending on the subtype of the question.

        Raises:
        -------
        ValueError
            If the subtype does not correspond to any known question type.
        """
        if subtype == "closest":
            closest_object = self._get_closest_object(color, objects_info)
            return closest_object["color_name"]
        elif subtype == "furthest":
            furthest_object = self._get_furthest_object(color, objects_info)
            return furthest_object["object_type"]
        elif subtype == "count":
            return self._get_count_of_objects_with_shape(color, objects_info)
        else:
            raise ValueError("Unknown question type")
        
    def _get_non_relational_answer(self, img_dim, subtype, color, objects_info):
        """
        @public

        Determine the answer to a non-relational question based on its subtype and reference color.

        Parameters:
        -----------
        img_dim : int
            The dimension (width and height) of the image.
        subtype : str
            The subtype of the question (e.g., "topbottom", "leftright", "shape").
        color : str
            The color of the reference object for the question.
        objects_info : list
            A list of dictionaries, where each dictionary contains information about an object 
            in the image, including its color, shape, and bounding box location.

        Returns:
        --------
        str
            The answer to the non-relational question. This could be "top", "bottom", "left", "right",
            or the shape of the object, depending on the subtype of the question.

        Raises:
        -------
        ValueError
            If the subtype does not correspond to any known question type.
        """
        if subtype == "topbottom":
            for obj_info in objects_info:
                # skip the question object
                if 'question' in obj_info:
                    continue
                if obj_info["color_name"] == color:
                    # consider the object to be on the top if its y-coordinate is less than half of the image height
                    # use the center of the bounding box to determine the y-coordinate
                    if self._compute_bbox_center(obj_info["bboxlocation"])[1] < img_dim // 2:
                        return "top"
                    else:
                        return "bottom"
        elif subtype == "leftright":
            for obj_info in objects_info:
                # skip the question object
                if 'question' in obj_info:
                    continue
                if obj_info["color_name"] == color:
                    # consider the object to be on the left if its x-coordinate is less than half of the image width
                    # use the center of the bounding box to determine the x-coordinate
                    if self._compute_bbox_center(obj_info["bboxlocation"])[0] < img_dim // 2:
                        return "left"
                    else:
                        return "right"
        elif subtype == "shape":
            for obj_info in objects_info:
                # skip the question object
                if 'question' in obj_info:
                    continue
                if obj_info["color_name"] == color:
                    return obj_info["object_type"]
        else:
            raise ValueError("Unknown question type")    
        
    def _add_image_description_matrix(self, objects_info):
        """
        @public
        
        Add an image description matrix to the objects' information.

        The matrix has a shape of (num_objects, num_attributes), where num_attributes = 6, 
        representing the color, shape, and 4 bounding box coordinates for each object.

        Parameters:
        -----------
        objects_info : list
            A list of dictionaries, where each dictionary contains information about an object 
            in the image, including its color, shape, and bounding box location.

        Returns:
        --------
        list
            The updated list of dictionaries with the image description matrix added for each object.
        """
        # create a matrix of shape (num_objects, num_attributes) 
        # where num_attributes = 6 (color, shape, 4x bbox)
        matrix = []
        for obj_info in objects_info:
            # skip the question object
            if 'question' in obj_info:
                continue
            # extend the bbox location to 4 elements
            bbox = obj_info["bboxlocation"][0] + obj_info["bboxlocation"][1]
            row = []
            row.append(obj_info["color_name"])
            row.append(obj_info["object_type"])
            row.extend(bbox)
            matrix.append(row)
        obj_info["image_description_matrix"] = matrix
        
        return objects_info    

    def generate_dataset(self, 
        img_dim=128,
        num_images=100, 
        min_objects=6, 
        max_objects=6, 
        visualize_bboxes=False, 
        add_noise=False, 
        noise_prob=0.05, 
        fixed_colors=True,
        fill_objects=True,
        reusable_palette_colors=False,
        shapes=None,
        palette=None):
        """
        Generate a dataset of images with various geometric shapes and save the details in a JSON file.

        Parameters:
        -----------
        img_dim : int, optional
            The dimension (width and height) of each generated image (default is 128).
        num_images : int, optional
            The number of images to generate (default is 100).
        min_objects : int, optional
            Minimum number of objects to include in each image (default is 0).
        max_objects : int, optional
            Maximum number of objects to include in each image (default is 6).
        visualize_bboxes : bool, optional
            Whether to draw bounding boxes around the shapes (default is True).
        add_noise : bool, optional
            Whether to add noise to the images (default is False).
        noise_prob : float, optional
            The probability of noise presence in the images (default is 0.05).
        fixed_colors : bool, optional
            Whether to use a fixed set of colors from the palette (default is True).
        fill_objects : bool, optional
            Whether the shapes should be filled (default is False).
        reusable_palette_colors : bool, optional
            Whether colors from the palette can be reused (default is False).
        shapes : dict, optional
            A dictionary of shape generation functions (default is a set of predefined shapes).
        palette : dict, optional
            A color palette for the shapes (default is a set of predefined colors).

        Returns:
        --------
        None
        """
        self.img_dim = img_dim

        if shapes is None:
            shapes = {
                "rectangle": self._gen_rectangle,
                "circle": self._gen_circle,
                "triangle": self._gen_triangle,
                "pentagon": self._gen_pentagon,
                "square": self._gen_square
            }
        
        if palette is None:
            palette = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'orange': (0, 165, 255),
                'pink': (243, 194, 246),
                'yellow': (0, 255, 255)
            }
        
        all_images_info = {}
        
        bar = progressbar.ProgressBar(maxval=100,
                                    widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                            progressbar.Percentage()])
        print('Data generation started.')
        bar.start()
        
        for i in range(1, num_images + 1):
            img, objects_info = self._generate_img(
                min_objects=min_objects, 
                max_objects=max_objects, 
                visualize_bboxes=visualize_bboxes, 
                add_noise=add_noise,
                noise_prob=noise_prob,
                fixed_colors=fixed_colors,
                fill_objects=fill_objects,
                shapes=shapes,
                palette=palette,
                reusable_palette_colors=reusable_palette_colors
            )

            if palette is not None and not reusable_palette_colors and fixed_colors and len(palette) == 6:
                objects_info = self._add_questions_and_answers(img_dim, objects_info, palette, shapes)
                objects_info = self._add_image_description_matrix(objects_info)
            
            path = os.path.join(self.base_path, f'{i}.png')
            cv2.imwrite(path, img)
            
            if i % (num_images / 100) == 0:
                bar.update(i / (num_images / 100))
            
            all_images_info[path] = objects_info
        
        json_path = os.path.join(self.base_path, 'descr.json')
        with open(json_path, 'w') as json_file:
            json.dump(all_images_info, json_file, cls=NumpyEncoder, indent=4)

        bar.finish()
        print('Data generation finished. Generated ' + str(num_images) + ' images and ' + str(10 * num_images) + ' relational and ' + str(10 * num_images) + ' non-relational questions.')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)        