
# 🍎🍌🍊 Fruits Classifier Team 2 SA4110

This is a CNN-Based Image Classifier for Fruit Recognition
## 🎯 Objective:
Develop a Convolutional Neural Network (CNN) model capable of accurately identifying and classifying images into four fruit categories:

- Apple 🍎
- Banana 🍌
- Orange 🍊
- Mixed Fruits (e.g., fruit salads or assorted fruit baskets)
## 🛠️ Getting started

This project uses [Python v3.9.21](https://www.python.org/downloads/release/python-3921/)

1. Clone repository

```
git clone https://github.com/Team-2-SA60/FruitClassifier.git
```

2. Open Terminal and change working directory
```
cd FruitClassifier
```

3. Install dependencies (pip install)
```
pip install -r requirements.txt
```

4. Run python script
```
python3 Machine_learning_CA_Project.py
```

5. Input command you want to run
```
Commands
1: Train new model
2: Evaluate saved best model
3: Predict individual image
4: Exit
```

For command 1: This command will start to train a new model
- In the .py file, you can edit the hyperparameters (set as global variables) accordingly.
- Set 'save_best_model = True' to enable saving the best model.

For command 2: This command will evaluate the test dataset using the best model trained previously ('bestmodel.pth').

For command 3: This command will predict and identify a single image of your choice.
- Place image in /predict/ folder (e.g /predict/download1.jpg)
- When asked to input name of file, input 'download1' to get a prediction

## 🧰 Tools & Technologies

- Python v3.9.21
- PyTorch v2.7.1
- torchvision v0.22.1
- scikit-learn v1.6.1
- pandas v2.3.0
- numpy v2.0.2

## Team 2️⃣

- [@Adrian](https://github.com/adriantlh)
- [@Bo Fei](https://github.com/Bofei2058)
- [@Cai Yun](https://github.com/vegecloud)
- [@Kin Seng](https://github.com/im-ksc)
- [@Gong Yuan](https://github.com/gongyuannn)
- [@Run Xin](https://github.com/ZRX471)
