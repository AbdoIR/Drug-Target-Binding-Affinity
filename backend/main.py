"""FastAPI Backend for Drug-Target Binding Affinity Prediction (KC-DTA)"""

import os
import sys
import logging
from pathlib import Path
from functools import lru_cache

import torch
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem.rdchem import ValenceType
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

ALLOWED_ORIGINS = ["*"]  # Allow all origins for local use

# Input validation limits (prevent DoS)
MAX_SMILES_LENGTH = 500
MAX_PROTEIN_LENGTH = 5000

# Add KCDTA to path
sys.path.insert(0, str(Path(__file__).parent.parent / "KCDTA"))
from models.cnn import cnn

# ============================================================================
# Pre-computed Constants
# ============================================================================

SEQ_VOC = "ACDEFGHIKLMNPQRSTVWXY"
SEQ_VOC_SET = frozenset(SEQ_VOC)  # Frozenset for O(1) lookup
L = 21  # len(SEQ_VOC) - hardcoded to avoid function call
AA_TO_IDX = {aa: idx for idx, aa in enumerate(SEQ_VOC)}

# Pre-compute atom symbol lookup (44 symbols)
_ATOM_SYMBOLS = ('C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 
                'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 
                'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 
                'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb')
ATOM_SYMBOL_IDX = {s: i for i, s in enumerate(_ATOM_SYMBOLS)}

# Pre-compute all 6 permutation index tuples for 3-mers
PERM_INDICES = ((0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0))

# Pre-allocated reusable tensors (will be set on startup with correct device)
_EMPTY_EDGE = None
_ZERO_Y = None

# ============================================================================
# Optimized Feature Extraction
# ============================================================================

@lru_cache(maxsize=50000)
def _atom_feat(symbol: str, degree: int, num_hs: int, valence: int, aromatic: bool) -> tuple:
    """Cached atom features - returns normalized 78-dim tuple."""
    feat = [0.0] * 78
    feat[ATOM_SYMBOL_IDX.get(symbol, 43)] = 1.0
    feat[44 + min(degree, 10)] = 1.0
    feat[55 + min(num_hs, 10)] = 1.0
    feat[66 + min(valence, 10)] = 1.0
    feat[77] = 1.0 if aromatic else 0.0
    s = sum(feat)
    return tuple(f / s for f in feat)


@lru_cache(maxsize=10000)
def smile_to_graph(smile: str) -> tuple:
    """Convert SMILES to molecular graph (cached). Returns (n_atoms, features, edges)."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")
    
    # Extract features using cached atom_feat
    features = tuple(
        _atom_feat(a.GetSymbol(), a.GetDegree(), a.GetTotalNumHs(), a.GetValence(ValenceType.IMPLICIT), a.GetIsAromatic())
        for a in mol.GetAtoms()
    )
    
    # Build edge list - flat tuple for faster tensor creation
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.extend((i, j, j, i))  # Both directions
    
    return len(features), features, tuple(edges)


@lru_cache(maxsize=2000)
def protein_features(seq: str) -> tuple:
    """Cached combined 2D+3D protein features. Returns (flat_2d, flat_3d)."""
    # 2D: Cartesian product of amino acid counts
    counts = [0] * L
    for c in seq:
        idx = AA_TO_IDX.get(c)
        if idx is not None:
            counts[idx] += 1
    
    flat_2d = tuple(counts[i] * counts[j] for i in range(L) for j in range(L))
    
    # 3D: K-mers with permutations
    pro_3d = [0.0] * (L * L * L)
    seq_len = len(seq)
    
    # Count trimers in one pass
    trimer_counts = {}
    for i in range(seq_len - 2):
        t = seq[i:i+3]
        trimer_counts[t] = trimer_counts.get(t, 0) + 1
    
    # Fill 3D matrix
    for trimer, count in trimer_counts.items():
        try:
            idx = tuple(AA_TO_IDX[c] for c in trimer)
        except KeyError:
            continue
        for p in PERM_INDICES:
            a, b, c = idx[p[0]], idx[p[1]], idx[p[2]]
            pro_3d[a * L * L + b * L + c] += count
    
    # Normalize
    max_val = max(pro_3d) if pro_3d else 0
    if max_val > 0:
        pro_3d = tuple(v / max_val for v in pro_3d)
    else:
        pro_3d = tuple(pro_3d)
    
    return flat_2d, pro_3d


def create_graph_data(smiles: str, protein_seq: str, device: torch.device) -> DATA.Data:
    """Create PyTorch Geometric Data object directly on device."""
    n_atoms, features, edges = smile_to_graph(smiles)
    flat_2d, flat_3d = protein_features(protein_seq)
    
    # Create tensors on device
    x = torch.tensor(features, dtype=torch.float32, device=device)
    
    if edges:
        edge_idx = torch.tensor(edges, dtype=torch.long, device=device).view(2, -1)
    else:
        edge_idx = _EMPTY_EDGE if _EMPTY_EDGE is not None and _EMPTY_EDGE.device == device else torch.empty((2, 0), dtype=torch.long, device=device)
    
    data = DATA.Data(x=x, edge_index=edge_idx, y=_ZERO_Y)
    data.dcpro = torch.tensor(flat_2d, dtype=torch.float32, device=device).view(1, L, L)
    data.target = torch.tensor(flat_3d, dtype=torch.float32, device=device).view(1, L, L, L)
    data.batch = torch.zeros(n_atoms, dtype=torch.long, device=device)
    
    return data


# ============================================================================
# FastAPI Application
# ============================================================================

from typing import Optional

class AppState:
    __slots__ = ('model', 'device', 'empty_edge', 'zero_y')  # Slots for memory efficiency
    def __init__(self):
        self.model: Optional[cnn] = None
        self.device: Optional[torch.device] = None
        self.empty_edge: Optional[torch.Tensor] = None
        self.zero_y: Optional[torch.Tensor] = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _EMPTY_EDGE, _ZERO_Y
    
    # Startup: Load model with optimizations
    model_path = Path(__file__).parent.parent / "KCDTA" / "model_cnn_kiba.model"
    state.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    state.model = cnn()
    state.model.load_state_dict(torch.load(model_path, map_location=state.device, weights_only=True))
    state.model.to(state.device)
    state.model.eval()  # Set to evaluation mode (disables dropout)
    
    # Freeze parameters for inference (additional optimization)
    for param in state.model.parameters():
        param.requires_grad = False
    
    # Pre-allocate reusable tensors
    _EMPTY_EDGE = torch.empty((2, 0), dtype=torch.long, device=state.device)
    _ZERO_Y = torch.zeros(1, device=state.device)
    
    logger.info(f"Model loaded on {state.device} with {sum(p.numel() for p in state.model.parameters()):,} parameters")
    yield
    
    # Shutdown
    state.model = None


app = FastAPI(
    title="Drug-Target Binding Affinity Prediction API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
# Local CORS: allow all
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    smiles: str = Field(..., min_length=1, max_length=MAX_SMILES_LENGTH, description="SMILES representation of the drug molecule")
    protein_sequence: str = Field(..., min_length=1, max_length=MAX_PROTEIN_LENGTH, description="Amino acid sequence of the target protein")


class PredictionResponse(BaseModel):
    smiles: str
    protein_sequence: str
    binding_affinity: float
    model_used: str = "KIBA"


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": state.model is not None, "device": str(state.device)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if state.model is None:
        raise HTTPException(503, "Model not loaded")
    
    smiles = request.smiles.strip()
    seq = request.protein_sequence.strip().upper()
    
    # Fast validation using pre-computed frozenset
    invalid_aa = set(seq) - SEQ_VOC_SET
    if invalid_aa:
        raise HTTPException(400, f"Invalid amino acids found: {invalid_aa}. Valid: {SEQ_VOC}")
    
    # Validate SMILES (this also caches valid molecules in RDKit)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(400, f"Invalid SMILES string: unable to parse molecule")
    
    # Additional molecule validation
    if mol.GetNumAtoms() == 0:
        raise HTTPException(400, "SMILES represents an empty molecule")
    
    try:
        data = create_graph_data(smiles, seq, state.device)
        
        with torch.inference_mode():
            affinity = state.model(data).item()
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(500, "Prediction failed due to internal error")
    
    return PredictionResponse(smiles=smiles, protein_sequence=seq, binding_affinity=round(affinity, 4))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        factory=False,
    )
