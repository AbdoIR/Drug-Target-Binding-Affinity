# Drug-Target Binding Affinity Prediction (KC-DTA, KIBA Model)

This repository provides a production-ready FastAPI backend and deep learning models for predicting drug-target binding affinity using the **KIBA model** from the KC-DTA (Knowledge-based Compound-protein interaction Drug-Target Affinity) project.

---

## 1. Project Overview

Drug-target binding affinity prediction is crucial for drug discovery. This project leverages deep learning (GCN, CNN, FNN) to predict the binding affinity between drug compounds (SMILES) and protein targets (amino acid sequences), using the KIBA dataset as a benchmark.

---

## 2. About the KC-DTA KIBA Model

- **KC-DTA**: Deep learning framework for compound-protein affinity prediction.
- **KIBA model**: Neural network trained on the KIBA dataset, which integrates multiple kinase inhibitor bioactivity measures into a single score.
- **Model features:**
  - Graph-based encoding for drugs (SMILES → molecular graph)
  - 3D/2D k-mer encoding for proteins
  - Hybrid neural architecture (GCN, CNN, FNN)
- Trained model used: `KCDTA/model_cnn_kiba.model`

---

## 3. Project Structure

```
.
├── backend                     # FastAPI backend
│   ├── main.py
│   └── requirements.txt
└── KCDTA/                      # Model code and scripts
    ├── models/
    │   └── cnn.py
    ├── data/                   # Training datasets
    │   ├── davis/
    │   └── kiba/
    ├── model_cnn_kiba.model    # Pre-trained kiba model
    ├── model_cnn_davis.model   # Pre-trained davis model
    ├── create_davis_kiba.py
    ├── training.py
    ├── test.py
    └── utils.py
```

---


## 4. Dataset & Data Preparation

- **KIBA and Davis datasets**: Provided in `KCDTA/data/` (raw formats)
- Data preprocessing and conversion scripts are available (see below for usage instructions)

### Data Conversion (PyTorch Format)

To convert the Davis or KIBA datasets into PyTorch format, run the data conversion script (if available in your local copy):

```bash
python KCDTA/create_davis_kiba.py
```

Note: The code provided is for the fifth fold in a five-fold cross-validation experiment.

The processed data will be saved in a directory (e.g., `KCDTA/data/processed/`).

---


## 5. Training & Evaluation

To train or evaluate the model, use the following commands:

```bash
cd KCDTA
python training.py   # Train the model
python test.py       # Evaluate the model
```

Arguments for `training.py`:
- The first argument: 0 for Davis, 1 for KIBA dataset.
- The second argument: 0 (use the CNN model provided).
- The third argument: CUDA index (e.g., 0 or 1). Adjust according to your system.

Example:
```bash
python training.py 0 0 0
```
This will train on the Davis dataset using the CNN model and CUDA device 0.

To test the trained model:
```bash
python test.py 0  # 0 for Davis, 1 for KIBA
```

Note: Only the fifth fold is provided by default.

---

## 6. Backend API (FastAPI)

### Features

- **/predict**: Predicts binding affinity for a given drug (SMILES) and protein (sequence)
- **/health**: Check API and model status
- CORS enabled for local development

### Requirements

- Python 3.11+
- See `backend/requirements.txt` for dependencies

### Running Locally

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The server will run at [http://localhost:8000](http://localhost:8000)

#### API Docs

- Interactive: [http://localhost:8000/docs](http://localhost:8000/docs)
- Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Example Usage

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "smiles": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
    "protein_sequence": "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKK"
})
print(response.json())
```

**Example Response:**

```json
{
  "smiles": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
  "protein_sequence": "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGS...",
  "binding_affinity": 10.6131,
  "model_used": "KIBA"
}
```

---

## A. Contributing & Support

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

For questions, contact the maintainer or open an issue on GitHub.

---

## B. References

- KC-DTA: [Original Paper](https://doi.org/10.1093/bioinformatics/btaa880)
- KIBA Dataset: [KIBA on Zenodo](https://zenodo.org/record/4032820)
- FastAPI: https://fastapi.tiangolo.com/

---

## C. License

This project is made for educational purposes and out of love for the developer community. Feel free to use, share, and learn from it ♥️.