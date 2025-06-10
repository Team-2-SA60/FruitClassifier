import os, torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Variables to change
# Hyperparameters for training
image_resize = 112
no_of_conv_layers = 3
conv1_out_channel = 40
learning_rate = 0.001
n_epochs = 200
early_stop_patience = 50
train_batch_size = 24
test_batch_size = 60

# Number of runs for training
n_runs = 1
save_best_model = False

# Directories
main_dir = os.getcwd()
train_dir = f"{main_dir}/train"
test_dir = f"{main_dir}/test"
predict_dir = f"{main_dir}/predict"

'''----------------------------------------------------------------------------------------------'''

# Fixed variables (DO NOT CHANGE)
# Flatten size (e.g. 28x28 image with 3 conv layers = 3 pooling layers, so divide 28 by 2 by 2 = 7, then flatten size = 7 x 7 x final_conv_out_channel)
flattened_size = image_resize
for i in range(no_of_conv_layers):
    flattened_size //= 2
flattened_size = flattened_size * flattened_size

# Final convolution layer out channel = 2 ** (Number of layers - 1) * conv1_out_channel
final_conv_out_channel = 2 ** (no_of_conv_layers - 1) * conv1_out_channel

'''----------------------------------------------------------------------------------------------'''

# Function to load data from directory
def prepare_data(target_dir):
    filepaths = []
    labels = []

    # Get filepaths of all images in dir
    images = os.listdir(target_dir)

    # Loop through all images to store each image paths as well as their labels
    for image in images:
        label = image.split("_")[0]                             # e.g. orange_41.jpg -> ["orange", "41.jpg"] -> "orange"
        if label not in ["orange", "apple", "mixed", "banana"]:
            continue
        filepaths.append(f"{target_dir}/{image}")               # e.g. /train/orange_41.jpg
        labels.append(label)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    return np.array(filepaths), torch.tensor(labels), le.classes_


# CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution layers
        # Start with in_channels=3 because RGB
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv1_out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channel, out_channels=conv1_out_channel * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=conv1_out_channel * 2, out_channels=conv1_out_channel * 4, kernel_size=3, padding=1)

        # Average Pooling Layer: downsample by a factor of 2.
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=flattened_size * final_conv_out_channel, out_features=final_conv_out_channel * 2)

        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=final_conv_out_channel * 2, out_features=4)

        # Activation function
        self.relu = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        # Apply convolution + ReLU + pooling
        x = self.pool(self.relu(self.conv1(x)))     # conv1
        x = self.pool(self.relu(self.conv2(x)))     # conv2
        x = self.pool(self.relu(self.conv3(x)))     # conv3

        # Flatten the feature maps
        x = x.view(-1, flattened_size * final_conv_out_channel)

        # Fully connected layers
        x = self.relu(self.fc1(x))                  # fc1
        x = self.dropout(x)                         # dropout layer

        # Output layer (out_features = 4)
        x = self.fc2(x)                             # fc2

        return x
  

# Function to perform image augmentation
def load_images(filepaths, purpose):
    # Instantiate class to transform image to tensor based on purpose
    transform = {
        "train": transforms.Compose(
            [
                transforms.RandomPerspective(),
                transforms.RandomResizedCrop((image_resize, image_resize), (0.5, 1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=(-0.01, 0.01)),
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2)),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value=0, inplace=False),
                v2.GaussianNoise(0, 0.05),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((image_resize,image_resize)),
                transforms.ToTensor(),
            ]
        )
    }

    image_tensors = []
    for path in filepaths:
        image = Image.open(path).convert("RGB")
        img_tensor = transform[purpose](image)
        image_tensors.append(img_tensor)

    batch_tensor = torch.stack(image_tensors)
    return batch_tensor


# Function to test model
def test(model, criterion, test_filepaths, test_labels, test_classes):
    # Set model to testing mode
    model.eval()

    test_loss, test_correct = 0, 0
    test_total_samples = len(test_filepaths)
    no_batch_tested = 0
    wrong_preds = []

    with torch.no_grad():
        for i in range(0, test_total_samples, test_batch_size):
            # Increase number of batches tested so far in this epoch
            no_batch_tested += 1

            # Load test images in batch to tensor
            batch_filepaths = test_filepaths[i : i + test_batch_size]
            batch_inputs = load_images(filepaths=batch_filepaths, purpose="test")
            batch_labels = test_labels[i : i + test_batch_size]

            # Forward pass: compute predicted outputs
            outputs = model(batch_inputs)

            # Get probability-distributions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)

            # Save and calculate some stats
            test_loss += criterion(outputs, batch_labels).item()
            test_correct += torch.sum(preds == batch_labels)  # compare predictions with labels

            # Save wrong predictions
            for j in range(len(preds)):
                if preds[j] != batch_labels[j]:
                    image_name = batch_filepaths[j].split("/")[-1]
                    wrong_preds.append([image_name, test_classes[preds[j]].item()])

    # Average test loss and test accuracy after epoch
    avg_test_loss = test_loss / no_batch_tested
    test_acc = test_correct / test_total_samples

    return avg_test_loss, test_acc, wrong_preds


# Function to train model
def train(model, criterion, optimizer, train_filepaths, train_labels, test_filepaths, test_labels, test_classes):

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_test_loss = float("inf")
    epochs_no_improvement = 0
    lr = learning_rate
    batch_size = train_batch_size

    for epoch in range(n_epochs):
        # Set model to training mode
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # For tracking and printing our training-progress
        train_total_samples = len(train_filepaths)
        train_loss, train_correct = 0, 0
        no_batch_trained = 0

        # Shuffle training data
        permutation = torch.randperm(train_total_samples)

        for i in range(0, train_total_samples, batch_size):
            # Increase number of batches trained so far in this epoch
            no_batch_trained += 1

            # Load training images in batch to tensor
            indices = permutation[i : i + batch_size]
            batch_filepaths = train_filepaths[indices]
            batch_inputs = load_images(filepaths=batch_filepaths, purpose="train")
            batch_labels = train_labels[indices]

            # Forward pass: compute predicted outputs
            outputs = model(batch_inputs)

            # Compute loss
            loss = criterion(outputs, batch_labels)
        
            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Get probability-distributions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)

            # Save and calculate some stats
            train_loss += loss.item()
            train_correct += torch.sum(preds == batch_labels)  # compare predictions with labels

        avg_train_loss = train_loss / no_batch_trained
        train_acc = train_correct / train_total_samples

        # Learning rate decay
        if avg_train_loss < 0.25:
            lr = 0.0001
        elif avg_train_loss < 0.5:
            lr = 0.0005
        else:
            lr = 0.001

        # Evaluate model using test data after training model for this epoch
        avg_test_loss, test_acc, wrong_preds = test(model, criterion, test_filepaths, test_labels, test_classes)

        # Logging
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc.item())
        history["test_loss"].append(avg_test_loss)
        history["test_acc"].append(test_acc.item())

        # Print stats
        print(f"Epoch {epoch + 1}\n",
            f"train loss = {avg_train_loss:.5f}, ",
            f"train accuracy = {train_acc * 100:.2f}%\n",
            f"test loss = {avg_test_loss:.5f}, ",
            f"test accuracy = {test_acc * 100:.2f}%")

        # Early Stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improvement = 0

            # Save best model (lowest test loss) if set to True
            if save_best_model:
                torch.save(model.state_dict(), f'{main_dir}/bestmodel.pth')
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement > early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history


# Functions to export to csv
def convert_df(data, parameter_name, parameter_value, n_run):
    df = pd.DataFrame(data)
    df.insert(0, parameter_name, parameter_value)
    df.insert(0, "run", n_run + 1)
    return df

def save_to_csv(dir, parameter_name, df: pd.DataFrame, tracker):
    output_dir = f"{dir}/output/{parameter_name}"
    output_path = f"{output_dir}/{tracker}.csv"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, mode="a", index=False, header=not os.path.exists(output_path))


# Model training logic
# (Load data > Train > Save training data and params)
def start_training():
    changed_parameter = input("Input changed parameter: ").strip()
    parameter_value = input("Input parameter value: ").strip()

    for i in range(n_runs):
        # Instantiate training model and define loss function + optimiser
        model = CNN()
        criterion = nn.CrossEntropyLoss()  # define loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Load training and testing data
        # Training data
        train_filepaths, train_labels, train_classes = prepare_data(train_dir)

        # Testing data
        test_filepaths, test_labels, test_classes = prepare_data(test_dir)

        # Train model
        history = train(model, criterion, optimizer, train_filepaths, train_labels, test_filepaths, test_labels, test_classes)

        # Save training loss/accuracy and test accuracy to csv
        # For each run, will automatically create directory and every subsequent runs will append into csv
        history_csv = convert_df(history, changed_parameter, parameter_value, n_run=i)
        save_to_csv(main_dir, changed_parameter, history_csv, "epoc_loss_acc")

        # Save parameters for current run
        params = [["image_resize", image_resize], ["no_of_conv_layers", no_of_conv_layers], ["conv1_out_channel", conv1_out_channel], ["learning_rate", learning_rate], ["n_epochs", n_epochs], ["batch_size", train_batch_size]]
        paramsdf = convert_df(params, changed_parameter, parameter_value, n_run=i)
        save_to_csv(main_dir, changed_parameter, paramsdf, "params")


# Predict individual test image
def predict(model, test_classes):
    # Set model to testing mode
    model.eval()

    with torch.no_grad():
        image_name = ""
        try:
            image_name = input("Input file name of image to predict: ")
            image_to_predict = [f"{predict_dir}/{image_name}.jpg"]
            batch_inputs = load_images(filepaths=image_to_predict, purpose="test")
        except:
            print("File name entered do not exist, only input name of file. E.g. 'banana_81'")
            return

        # Forward pass: compute predicted outputs
        outputs = model(batch_inputs)

        # Get probability-distributions
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, dim=1)

        print(f"Prediction for '{image_name}.jpg': {test_classes[preds[0]]}")


# Manually test best model
def eval_best_model(purpose):    
    # Test best model (Manually)
    best_model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=learning_rate)

    # Load saved best model
    best_model.load_state_dict(torch.load(f"{main_dir}/bestmodel.pth"))

    # Testing data
    test_filepaths, test_labels, test_classes = prepare_data(test_dir)

    if purpose == "eval":
        avg_test_loss, test_acc, wrong_preds = test(best_model, criterion, test_filepaths, test_labels, test_classes)
        print(f"Final test loss = {avg_test_loss:.5f}, test accuracy = {test_acc * 100:.2f}%, wrong predictions = {wrong_preds}")
    elif purpose == "predict":
        predict(best_model, test_classes)


# Main program
def main():
    while True:
        print("", "Commands", "1: Train new model", "2: Evaluate saved best model", "3: Predict individual image", "4: Exit", "", sep="\n")

        cmd = input("Input command: ")

        if cmd == "1":
            print("Training new model according to specified global parameters")
            start_training()
        elif cmd == "2":
            print("Evaluating saved best model with test dataset")
            eval_best_model(purpose="eval")
        elif cmd == "3":
            eval_best_model(purpose="predict")
        elif cmd == "4":
            return
        else:
            print("Please input 1-3 only")

        input("\nPress 'Enter' to continue")


if __name__ == "__main__":
    main()