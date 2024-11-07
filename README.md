Here’s a simple `README.md` for your project:

# Beach vs. Mountain Classifier

This project implements a binary classifier to distinguish between beach and mountain images using a modified ResNet50 model pre-trained on the Places365 dataset. The classifier is trained using a custom dataset and evaluated to measure its accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Results](#results)

## Project Overview
The classifier leverages transfer learning with ResNet50 pre-trained on the Places365 dataset, which is then fine-tuned for a binary classification task (beach vs. mountain). After training, the model can be used to classify new images.

## Dataset
The model is trained on the custom `BeachMountainDataset`, located in the `dataset.py` file. This dataset is expected to contain labeled images of beaches and mountains in the specified directory.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/beach-mountain-classifier.git
   cd beach-mountain-classifier
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that the `resnet50_places365.pth.tar` file (pre-trained Places365 weights) is in the project directory. You can download it from the [official Places365 repository](https://github.com/CSAILVision/places365).

## Training the Model
To train the model, simply run:
```bash
python train.py
```

This will:
- Train the model for 10 epochs on the custom dataset.
- Save the trained model weights in `beach_mountain_classifier.pth`.
- Generate and save training/validation loss and accuracy plots as `training_results.png`.

## Evaluating the Model
To evaluate the model, load the saved weights and run inference on a test set. Example code for loading and testing the model is provided in `train.py`. Alternatively, you can add the following code to evaluate the model after training:

```python
# Load the model
model = Places365Classifier(num_classes=2)
model.load_state_dict(torch.load('beach_mountain_classifier.pth'))
model.eval()

# Use `test_loader` to evaluate accuracy on test images
```

## Results
Training and validation results are saved as `training_results.png`. The accuracy of the model on test images is printed at the end of evaluation.

## Notes
- This code uses CUDA if available; otherwise, it defaults to CPU.
- The `torch.load` function now includes `weights_only=True` to prevent security risks from untrusted models.

## License
This project is licensed under the MIT License.
