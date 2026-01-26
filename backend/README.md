# Drug-Target Binding Affinity Prediction API (KC-DTA, KIBA Model)

This repository provides a production-ready FastAPI backend for predicting drug-target binding affinity using the **KIBA model** from the KC-DTA (Knowledge-based Compound-protein interaction Drug-Target Affinity) project.

---

## About the KIBA Model (KC-DTA)

- **KC-DTA** is a deep learning framework for predicting the binding affinity between drug compounds (SMILES) and protein targets (amino acid sequences).
- The **KIBA model** is a neural network trained on the KIBA dataset, a benchmark for kinase inhibitor bioactivity, combining multiple bioactivity measures into a single score.
- The model uses:
  - Graph-based molecular encoding for drugs (SMILES → molecular graph)
  - 3D/2D k-mer encoding for proteins
  - A hybrid neural architecture (GCN, CNN, FNN)
- The trained model file is `KCDTA/model_cnn_kiba.model`.

---

## Backend API (FastAPI)

### Features

- **/predict** endpoint: Predicts binding affinity for a given drug (SMILES) and protein (sequence)
- **/health** endpoint: Check API and model status
- CORS enabled for local development
- Highly optimized for speed and memory

### Requirements

- Python 3.11+
- See `requirements.txt` for dependencies

### Running Locally

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API**

   ```bash
   python main.py
   ```

   The server will run at [http://localhost:8000](http://localhost:8000)

3. **API Docs**
   - Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Example Usage

#### Predict Binding Affinity (Python)

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "smiles": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
    "protein_sequence": "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKK"
})
print(response.json())
```

#### Example Response

```json
{
  "smiles": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
  "protein_sequence": "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGS...",
  "binding_affinity": 10.6131,
  "model_used": "KIBA"
}
```

---

## Project Structure

```
KCDTA/                  # Model code and weights
  model_cnn_kiba.model  # Trained KIBA model
Backend/
  main.py               # FastAPI backend
  requirements.txt      # Python dependencies
  Dockerfile            # (optional) for container deployment
```

---

## Docker (Optional)

Build and run the API in a container:

```bash
docker build -t kc-dta-api .
docker run -p 8000:8000 kc-dta-api
```

---

## References

- KC-DTA: [Original Paper](https://doi.org/10.1093/bioinformatics/btaa880)
- KIBA Dataset: [KIBA on Zenodo](https://zenodo.org/record/4032820)
- FastAPI: https://fastapi.tiangolo.com/

---

## License

This project is for research and educational use. See LICENSE for details.
