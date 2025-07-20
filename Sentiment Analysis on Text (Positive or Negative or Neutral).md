# **Sentiment Analysis (Positive / Neutral / Negative)**

---
### **Tool:** Google Colab

### **Library:** `transformers` (HuggingFace)

### **Model Used:** `cardiffnlp/twitter-roberta-base-sentiment` (Trained on 3 classes: Positive, Neutral, Negative)

---
## **1. Setup Google Colab (No Installation Needed)**

1. Go to Google Colab.
    
2. Click **“New Notebook”**.
    
3. Make sure runtime is GPU (optional but faster):
    
    - `Runtime → Change runtime type → Hardware accelerator → T4 GPU`.

---

## **2. Install & Import Libraries**

👉 **Copy & run this cell:**

```python
!pip install transformers

from transformers import pipeline
```

✔ **What happens now?**

- Installs HuggingFace’s **Transformers** library.
    
- Imports the **pipeline** function (a simple high-level API to use pre-trained models).
    
---

## **3. Load the 3-Class Sentiment Model**

👉 **Run this cell:**

```python
# Load a pre-trained model for 3-class sentiment (Positive, Neutral, Negative)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# Mapping model labels to human-readable form
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

print("✅ 3-Class Sentiment Model Ready!")
```

✔ **What happens now?**

- Downloads and loads the **Twitter RoBERTa model**, which can classify text into **Positive, Neutral, or Negative**.
    
- Creates a **mapping** so instead of showing `LABEL_0`, you see `Negative / Neutral / Positive`.
---

## **4. Enter a Sentence & Get Sentiment**

👉 **Run this cell:**

```python
text = input("Enter a sentence: ")
result = sentiment_analyzer(text)[0]
print(f"✅ Sentiment: {label_map[result['label']]} (Confidence: {result['score']*100:.2f}%)")
```

✔ **What happens now?**

1. It will **ask you to type any sentence** (e.g., _“The movie was fantastic!”_).
    
2. The model will instantly predict the **sentiment** (Positive, Neutral, or Negative) along with **confidence score**.

---
## 5. **Sample Outputs**

1. **Input:** `"The service was terrible"`
```php
✅ Sentiment: Negative (Confidence: 97.26%)
```

2. **Input:** `"The phone works as expected"`
```php
✅ Sentiment: Positive (Confidence: 72.76%)
```

3. **Input:** `"This course is amazing!"`
```php
✅ Sentiment: Positive (Confidence: 99.04%)
```

---

	