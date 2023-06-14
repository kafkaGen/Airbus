class Config:
    # Training
    epochs = 15
    learning_rate = 2e-3
    num_classes = 1
    loss_alpha = 0.75
    threshold = 0.85
    
    # Data preprocessing
    images_path = 'data/train_v2'
    masks_path = 'data/train_ship_segmentations_v2.csv'
    test_path = 'data/test_v2'
    filtered_masks_path = 'data/masks.csv'
    hue_path = 'data/hue.csv'
    labels_path = 'data/labels.csv'
    model_path = 'models/'
    batch_size = 16
    buffer_size = 1000
    random_state = 17
    
    # Data transformations
    channels = (64, 128, 256, 512)
    input_shape = (None, None, 3)
    