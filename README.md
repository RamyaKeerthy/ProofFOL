# ProofFOL: Improving NL-to-FOL Translation of  LMs

We introduce ProofFOL, a high-quality FOL-annotated dataset containing $10.4k$ examples, with multiple premises and a conclusion. This is used to perform a finetuning diagnosis for multiple small language models.

## 📦 Installation
1. Clone the repository:
 ```
 git clone https://github.com/anonymous-link
```

3. Install dependencies:
```
pip install -r requirements.txt
```
---

## 📂 Data
- All **data** required for this project is located in the [`data/`](data/) directory.  
- **Test datasets** are exactly as described in our accompanying paper (please refer to the paper for detailed dataset descriptions).  

---

## Fine-tuning the Model
To fine-tune the model on your dataset, use the provided script:

```
python scripts/finetune.py
```

---

## Extracting Inferences
### Few-Shot Inference
Run few-shot inferences using:

```
bash inference-few-shot.sh
```

Prompts: All required prompts and instructions are located in the [`prompts/`](prompts/) directory. Review and edit these prompts for your specific use case.

### Fine-tuned Inference
Run fine-tuned inferences using:

```
bash inference-fine-tuned.sh
```

---

## Evaluating Inferences
Evaluation is done in two steps:
1. Pass model outputs to the inference tool:
```
python tool_inference.py
```
3. Calculate metrics & error distribution:
```
python error_distribution.py
```

These scripts will provide quantitative metrics and a detailed error distribution analysis for your outputs as mentioned in our paper.
