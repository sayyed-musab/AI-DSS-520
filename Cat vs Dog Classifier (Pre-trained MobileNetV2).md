# **Step-by-Step: Cat vs Dog Classifier (Pre-trained MobileNetV2)**

### ## **1. Setup Google Colab (No Installation Needed)**

1. Go to Google Colab.
    
2. Click **“New Notebook”**.
    
3. Make sure runtime is GPU (optional but faster):
    
    - `Runtime → Change runtime type → Hardware accelerator → T4 GPU`.
        

---

## ## **2. Install & Import Libraries**

👉 **Copy-Paste this into the first cell and run it (Shift + Enter):**

```python
!pip install tensorflow

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from google.colab import files
from PIL import Image

```

✔ **What happens now?**

- Installs TensorFlow.
    
- Loads all required libraries to handle the model, process images, and upload files.

---

### **3. Load Pre-trained Model**

👉 **Run this:**

```python
model = MobileNetV2(weights='imagenet')
print("✅ Cat vs Dog Model Ready!")
```

✔ **What happens now?**

- Loads a **pre-trained MobileNetV2 model** (already trained on 1,000 ImageNet classes).
    
- Prints a message once ready.


✔ **Why MobileNetV2?**

- Lightweight & fast.
    
- Already trained on **1,000 classes** (cats, dogs, fruits, etc.).

---

### **4. Upload Image & Predict Cat or Dog**

👉 **Run this cell:**

```python
uploaded = files.upload()

# ImageNet labels for cats and dogs
cat_labels = ["tabby", "tiger_cat", "Persian_cat", "Siamese_cat", 
              "Egyptian_cat", "lynx", "snow_leopard", "leopard", 
              "cougar", "tiger"]
dog_labels = ["Chihuahua", "Japanese_spaniel", "Maltese_dog", "Pekinese", 
              "Shih-Tzu", "Blenheim_spaniel", "papillon", "toy_terrier", 
              "Rhodesian_ridgeback", "beagle", "bloodhound", "Walker_hound", 
              "English_foxhound", "redbone", "borzoi", "Irish_wolfhound", 
              "Italian_greyhound", "whippet", "Ibizan_hound", "Norwegian_elkhound",
              "otterhound", "Saluki", "Scottish_deerhound", "Weimaraner", "Staffordshire_bullterrier",
              "American_Staffordshire_terrier", "Bedlington_terrier", "Border_terrier",
              "Kerry_blue_terrier", "Irish_terrier", "Norfolk_terrier", "Norwich_terrier",
              "Yorkshire_terrier", "wire-haired_fox_terrier", "Lakeland_terrier",
              "Sealyham_terrier", "Airedale", "cairn", "Australian_terrier", 
              "Dandie_Dinmont", "Boston_bull", "miniature_schnauzer", "giant_schnauzer",
              "standard_schnauzer", "Scotch_terrier", "Tibetan_terrier", "silky_terrier",
              "soft-coated_wheaten_terrier", "West_Highland_white_terrier", "Lhasa", 
              "flat-coated_retriever", "curly-coated_retriever", "golden_retriever",
              "Labrador_retriever", "Chesapeake_Bay_retriever", "German_short-haired_pointer",
              "vizsla", "English_setter", "Irish_setter", "Gordon_setter", "Brittany_spaniel",
              "clumber", "English_springer", "Welsh_springer_spaniel", "cocker_spaniel",
              "Sussex_spaniel", "Irish_water_spaniel", "kuvasz", "schipperke", "groenendael",
              "malinois", "briard", "kelpie", "komondor", "Old_English_sheepdog", "Shetland_sheepdog",
              "collie", "Border_collie", "Bouvier_des_Flandres", "Rottweiler", "German_shepherd",
              "Doberman", "miniature_pinscher", "Greater_Swiss_Mountain_dog", "Bernese_mountain_dog",
              "Appenzeller", "EntleBucher", "boxer", "bull_mastiff", "Tibetan_mastiff", "French_bulldog",
              "Great_Dane", "Saint_Bernard", "Eskimo_dog", "malamute", "Siberian_husky", "affenpinscher",
              "basenji", "pug", "Leonberg", "Newfoundland", "Great_Pyrenees", "Samoyed", "Pomeranian",
              "chow", "keeshond", "Brabancon_griffon", "Pembroke", "Cardigan", "toy_poodle", 
              "miniature_poodle", "standard_poodle"]

for fn in uploaded.keys():
    # Load & preprocess image
    img = image.load_img(fn, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

    # Check if any predicted label belongs to Cat or Dog list
    prediction_label, confidence = None, 0
    for (imagenet_id, label, score) in decoded:
        if label in cat_labels:
            prediction_label, confidence = "Cat", score
            break
        elif label in dog_labels:
            prediction_label, confidence = "Dog", score
            break

    if prediction_label:
        print(f"✅ It's a {prediction_label}! (Confidence: {confidence*100:.2f}%)")
    else:
        print("❌ Not clearly a Cat or Dog.")
```

✔ **What happens now?**

1. It will **prompt you to upload an image** (click “Choose File”).
    
2. The model will check top-5 predictions and compare them with the **cat & dog label lists**.
    
3. It will print:
    
    - ✅ **“It’s a Cat!”** (with confidence %) if it detects a cat.
        
    - ✅ **“It’s a Dog!”** (with confidence %) if it detects a dog.
        
    - ❌ **“Not clearly a Cat or Dog.”** for other objects.

---

## **5. Sample Output**

- If you upload a **cat image**:
```rust
✅ It's a Cat! (Confidence: 87.45%)
```

- If you upload a **dog image**:
```rust
✅ It's a Dog! (Confidence: 92.13%)
```

- If you upload something else (apple, car, etc.):
```css
❌ Not clearly a Cat or Dog.
```

---
## **6. Run Again for Another Image**

Just re-run the **Upload an Image** cell and upload a new image.

---


