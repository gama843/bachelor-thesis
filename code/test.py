import torch

def test_coordinate_tagging():
    # Fake a small batch of CNN feature maps (simpler than running through a real CNN)
    batch_size = 3
    channels = 256
    height, width = 5, 5  # Small feature map for easy visualization
    
    # Create fake feature maps with random values
    feature_maps = torch.rand(batch_size, channels, height, width)
    print(f"Feature maps shape: {feature_maps.shape}")
    
    # Create normalized coordinate meshgrid in range [-1, 1]
    y_coords = torch.linspace(-1, 1, height).unsqueeze(1).expand(height, width)
    x_coords = torch.linspace(-1, 1, width).expand(height, width)
    
    print(f"y_coords shape: {y_coords.shape}")
    print(f"x_coords shape: {x_coords.shape}")
    
    # Print the actual coordinate values for inspection
    print("\ny_coords values:")
    print(y_coords)
    print("\nx_coords values:")
    print(x_coords)
    
    # Reshape coordinates to match feature map dimensions
    coords = torch.stack((y_coords, x_coords), dim=0)  # [2, height, width]
    print(f"Stacked coords shape: {coords.shape}")
    
    coords = coords.unsqueeze(0).expand(batch_size, 2, height, width)
    print(f"Expanded coords shape: {coords.shape}")
    
    # Concat feature maps and coordinates along the channel dimension
    feature_maps_with_coords = torch.cat([feature_maps, coords], dim=1)
    print(f"Feature maps with coords shape: {feature_maps_with_coords.shape}")
    
    # Reshape to get a set of objects
    # [batch_size, channels+2, height, width] -> [batch_size, height*width, channels+2]
    objects = feature_maps_with_coords.permute(0, 2, 3, 1).reshape(batch_size, height*width, channels+2)
    print(f"Final objects shape: {objects.shape}")
    
    # Verify that the last two elements of each object vector are indeed coordinates
    # Extract one object from the first batch item
    sample_object = objects[0, 12]  # Middle object (12th of 25)
    print(f"\nSample object length: {len(sample_object)}")
    print(f"Sample object coordinates (last two values): {sample_object[-2:].tolist()}")
    
    # Verify these coordinates match the original meshgrid
    center_y = y_coords[2, 2].item()  # Middle of 5x5 grid
    center_x = x_coords[2, 2].item()
    print(f"Expected coordinates at position (2,2): [{center_y}, {center_x}]")
    
    # Verify that channels are preserved correctly
    # Check if feature values match before and after transformation
    original_features = feature_maps[0, :, 2, 2]  # All channels at position (2,2)
    transformed_features = objects[0, 12, :-2]    # All channels of 12th object except coords
    
    features_match = torch.allclose(original_features, transformed_features)
    print(f"Feature values preserved correctly: {features_match}")

if __name__ == "__main__":
    test_coordinate_tagging()