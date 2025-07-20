# **Step-by-Step Guide: Build a Simple Image Classifier**

## **1. Setup Google Colab (No Installation Needed)**

1. Go to Google Colab.
    
2. Click **“New Notebook”**.
    
3. Make sure runtime is GPU (optional but faster):
    
    - `Runtime → Change runtime type → Hardware accelerator → T4 GPU`.
        

---

## **2. Install & Import Libraries**

👉 **Copy-Paste this into the first cell and run it (Shift + Enter):**

```python
# Install libraries (if not already installed)
!pip install tensorflow

# Import required libraries
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from google.colab import files
from PIL import Image
```

✔ **What happens now?**

- Installs TensorFlow.
    
- Loads all required libraries to handle the model, process images, and upload files.
---

## **3. Load Pre-trained Model**

👉 **Run this cell:**

```python
# Load the pre-trained MobileNetV2 model (ImageNet trained)
model = MobileNetV2(weights='imagenet')
print("✅ Model Loaded Successfully!")
```

✔ **What happens now?**

- Loads a **pre-trained MobileNetV2 model** (already trained on 1,000 ImageNet classes).
    
- Prints a message once ready.


✔ **Why MobileNetV2?**

- Lightweight & fast.
    
- Already trained on **1,000 classes** (cats, dogs, fruits, etc.).

---

## **4. Upload an Image**

👉 **Run this cell:**

```python
# Upload an image
uploaded = files.upload()

for fn in uploaded.keys():
    # Load the uploaded image
    img_path = fn
    img = image.load_img(img_path, target_size=(224, 224))  # resize for MobileNet

    # Convert image to array & preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    print("✅ Prediction Completed!")

    # Decode top-3 predictions
    for i, (imagenet_id, label, score) in enumerate(decode_predictions(preds, top=3)[0]):
        print(f"{i+1}. {label} ({score*100:.2f}%)")
```

✔ **What happens now?**

- It will **prompt you to upload any image (cat, dog, apple, car, etc.)**.
    
- Model will output **Top 3 predictions with confidence scores**.

---

## **5. Sample Output**

If you upload a cat picture:

```scss
✅ Prediction Completed!
1. tabby (85.23%)
2. tiger_cat (10.12%)
3. Egyptian_cat (3.45%)
```

---

## **6. Run Again for Another Image**

Just re-run the **Upload an Image** cell and upload a new image.

---

