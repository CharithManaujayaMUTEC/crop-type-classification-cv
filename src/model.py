from tensorflow.keras import layers, models


# =============================================================================
# Build CNN Model
# =============================================================================

def build_model(num_classes):
    """
    Build and compile a Convolutional Neural Network (CNN) for
    EuroSAT land-cover image classification.

    Parameters
    ----------
    num_classes : int
        Total number of output classes.

    Returns
    -------
    tensorflow.keras.Model
        Compiled CNN model.
    """

    # Create a Sequential CNN model
    model = models.Sequential([

        # -----------------------------------------------------------------
        # First Convolutional Block
        # -----------------------------------------------------------------

        # Apply 32 convolution filters of size 3×3
        layers.Conv2D(
            32,
            (3,3),
            activation='relu',
            input_shape=(64,64,3)
        ),

        # Reduce feature map size using max pooling
        layers.MaxPooling2D(2,2),

        # -----------------------------------------------------------------
        # Second Convolutional Block
        # -----------------------------------------------------------------

        # Apply 64 convolution filters
        layers.Conv2D(
            64,
            (3,3),
            activation='relu'
        ),

        # Downsample the feature maps
        layers.MaxPooling2D(2,2),

        # -----------------------------------------------------------------
        # Fully Connected Layers
        # -----------------------------------------------------------------

        # Convert feature maps into a one-dimensional vector
        layers.Flatten(),

        # Hidden dense layer with 128 neurons
        layers.Dense(
            128,
            activation='relu'
        ),

        # Output layer with Softmax activation
        # Produces probabilities for each land-cover class
        layers.Dense(
            num_classes,
            activation='softmax'
        )
    ])

    # ---------------------------------------------------------------------
    # Compile the Model
    # ---------------------------------------------------------------------

    model.compile(

        # Optimization algorithm
        optimizer='adam',

        # Loss function for multi-class classification
        loss='categorical_crossentropy',

        # Evaluation metric
        metrics=['accuracy']
    )

    # Return the compiled CNN model
    return model