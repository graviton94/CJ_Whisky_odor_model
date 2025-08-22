# CJ Whisky Odor & Taste Prediction Model

🥃 **Advanced Whisky Sensory Prediction System with Phase 3 Concentration Adaptivity**

## 🌟 Features

### Core Capabilities
- **Multi-Modal Prediction**: Odor and taste descriptor prediction
- **Phase 3 Concentration Adaptivity**: Biological realism with S-curve concentration response
- **Enhanced Ontology System**: 67 synergy + functional group rules with adaptive strengths
- **Expert Feedback Learning**: Real-time model improvement from sensory evaluations
- **AI Tasting Notes**: GPT-powered whisky description generation

### Technical Highlights
- **Weber-Fechner Logarithmic Scaling** (Phase 2)
- **Dynamic Concentration Adaptivity Engine** (Phase 3)
- **Amplification Control System** (±50% limits)
- **2048-dimensional Molecular Features**
- **Streamlit Web Interface**

## 🏗️ Architecture

```
📦 CJ_Whisky_odor_model
├── 🤖 Model Files
│   ├── odor_finetune.pth          # Fine-tuned odor prediction model
│   ├── taste_finetune.pth         # Fine-tuned taste prediction model
│   └── model_state.json           # Model versioning & metadata
│
├── 🧠 Core System
│   ├── concentration_adaptivity.py # Phase 3: S-curve concentration engine
│   ├── ontology_manager.py        # Enhanced ontology with 67+ rules
│   ├── integrated_system.py       # Unified prediction pipeline
│   └── gui_enhanced_clean.py      # Main Streamlit interface
│
├── 🔮 Prediction Engines
│   ├── predict_hybrid.py          # Hybrid MLP + ontology prediction
│   ├── predict_finetuned.py       # Fine-tuned model prediction
│   └── predict.py                 # Basic prediction (updated)
│
├── 📊 Data Processing
│   ├── data_processor.py          # Feature extraction & preprocessing
│   ├── description_generator.py   # AI-powered tasting notes
│   └── retrain_models.py          # Expert feedback learning
│
└── 📁 Data & Config
    ├── sample_gcms_input.csv       # Example GC-MS input
    ├── ontology_rules_*.json       # Synergy & masking rules
    └── src/data/                   # Training datasets
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision streamlit rdkit pandas numpy scikit-learn openai
```

### Run the Application
```bash
streamlit run src/gui_enhanced_clean.py
```

### Basic Prediction Example
```python
from src.integrated_system import IntegratedPredictionSystem

# Initialize system
system = IntegratedPredictionSystem()

# Input molecules with concentrations
input_molecules = [
    {"SMILES": "CCO", "peak_area": 1000.0},           # Ethanol
    {"SMILES": "CC(=O)OCC", "peak_area": 500.0}       # Ethyl acetate
]

# Predict with Phase 3 concentration adaptivity
result = system.predict_mixture(
    input_molecules=input_molecules,
    mode='both',  # 'odor', 'taste', or 'both'
    use_ontology=True,
    confidence_threshold=0.7
)

print(f"Odor: {result['prediction']['odor']['corrected']}")
print(f"Taste: {result['prediction']['taste']['corrected']}")
```

## 🧪 Phase Development History

### Phase 1: Foundation
- ✅ Basic MLP models for odor/taste prediction
- ✅ Static ontology rules implementation
- ✅ Feature extraction pipeline

### Phase 2: Weber-Fechner Integration
- ✅ Logarithmic scaling for biological realism
- ✅ Controlled amplification (7-15% vs previous 30-50%)
- ✅ Enhanced rule strength management

### Phase 3: Concentration Adaptivity
- ✅ S-curve concentration response modeling
- ✅ Dynamic threshold calculation per functional group
- ✅ Mixture complexity analysis with Shannon entropy
- ✅ Amplification limits (±50% maximum change)
- ✅ Biological sensory threshold integration

## 📊 Performance Metrics

### Prediction Accuracy
- **Odor Descriptors**: 12 key descriptors (Fruity, Sweet, Woody, etc.)
- **Taste Descriptors**: 7 key descriptors (Sweet, Bitter, Sour, etc.)
- **Concentration Adaptivity**: 85% average strength reduction for biological realism
- **Rule Coverage**: 67+ synergy and functional group rules

### Phase 3 Improvements
- **Over-amplification Control**: Fruity 122% → 50% (Phase 3 fix)
- **Concentration Response**: S-curve modeling with steepness=8.0
- **System Stability**: 44 simultaneous rule applications without conflicts

## 🔬 Molecular Features

### Physical Properties
- Molecular weight, LogP, polar surface area
- Rotatable bonds, H-bond donors/acceptors
- Aromatic/aliphatic ring counts

### Functional Groups (11 types)
- **Alcohol, Ester, Aldehyde**: Primary flavor contributors
- **Terpene, Phenol, Furan**: Complexity enhancers
- **Sulfur, Pyrazine, Lactone**: Specialty compounds
- **Fatty Chain, Amine**: Supporting groups

### Concentration Features
- Individual molecular concentrations
- Relative concentration ratios
- Mixture complexity scores
- Primary/secondary component ratios

## 🎯 Ontology System

### Synergy Rules (Enhanced)
```python
# Example: Fruity enhancement
Triggers: ['ester', 'alcohol'] 
Effect: Base ×1.08 → Adaptive ×1.01-1.02 (concentration dependent)
Threshold: 0.20 concentration minimum
```

### Functional Group Effects
```python
# Example: Ester group effects
Fruity: ×1.06 → ×1.013 (adaptive)
Sweet: ×1.04 → ×1.009 (adaptive)
Citrus: ×1.03 → ×1.007 (adaptive)
```

### Masking Rules
- Sulfur compounds: General sensory suppression
- High fatty acid concentrations: Sweet/floral masking
- Phenolic compounds: Selective descriptor interference

## 🤖 AI Integration

### GPT-Powered Features
- **Tasting Notes Generation**: Professional whisky descriptions
- **Bilingual Output**: English and Korean tasting notes
- **Context-Aware**: Based on predicted descriptor profiles

### Expert Learning System
- **Real-time Feedback**: Sensory expert input integration
- **Model Retraining**: Automated fine-tuning from expert data
- **Similarity Analysis**: Mixture composition comparison
- **Performance Tracking**: Learning effectiveness monitoring

## 📁 File Structure

### Core Files (Production Ready)
```
✅ KEEP - Core functionality
🗑️ REMOVED - Development/testing files (67 files cleaned)

src/
├── ✅ gui_enhanced_clean.py      # Main interface
├── ✅ concentration_adaptivity.py # Phase 3 engine
├── ✅ ontology_manager.py        # Rule system
├── ✅ integrated_system.py       # Unified pipeline
├── ✅ predict_hybrid.py          # Hybrid prediction
├── ✅ predict_finetuned.py       # Fine-tuned models
├── ✅ description_generator.py   # AI descriptions
├── ✅ retrain_models.py          # Expert learning
└── ✅ data_processor.py          # Feature extraction

Models/
├── ✅ odor_finetune.pth         # Production odor model
├── ✅ taste_finetune.pth        # Production taste model
└── ✅ model_state.json          # Version control

Data/
├── ✅ sample_gcms_input.csv     # Example input
├── ✅ ontology_rules_odor.json  # Odor rules
└── ✅ ontology_rules_taste.json # Taste rules
```

## 🛠️ Development

### Setup Development Environment
```bash
git clone <your-repo>
cd CJ_Whisky_odor_model
pip install -r requirements.txt
```

### Training New Models
```python
from src.train_finetune import train_finetuned_model

# Train with expert data
train_finetuned_model(
    mode='odor',  # or 'taste'
    expert_data_path='src/data/mixture_trials_learn.jsonl',
    epochs=100
)
```

### Adding New Ontology Rules
```python
from src.ontology_manager import OntologyManager

manager = OntologyManager()
# Add custom synergy rule
manager.synergy_rules["NewDescriptor"] = {
    "triggers": [["functional_group1", "functional_group2"]],
    "strength": 1.1,
    "conditions": {"min_concentration": 0.15}
}
manager.save_rules("custom_rules.json")
```

## 📈 Future Roadmap

### Potential Phase 4 Options
- **Dynamic Rule Learning**: Automated pattern discovery from data
- **Temporal Modeling**: Aging and oxidation effect modeling  
- **Personal Preferences**: Individual sensory difference modeling
- **Real-time QC**: Production monitoring integration
- **Multi-sensory**: Visual, tactile, and thermal integration

## 🤝 Contributing

### Code Standards
- Python 3.8+ compatibility
- Type hints for all functions
- Comprehensive logging
- Unit tests for core functions

### Adding Features
1. Fork the repository
2. Create feature branch (`feature/new-feature`)
3. Add tests and documentation
4. Submit pull request

## 📄 License

This project is proprietary software developed for CJ Group.

## 🏆 Acknowledgments

- **CJ Whisky Research Team**: Domain expertise and sensory evaluation
- **Phase 3 Development**: Biological realism and concentration adaptivity
- **Production Optimization**: 67-file cleanup for maintainable codebase

---

**Status**: ✅ Production Ready | **Version**: Phase 3 Complete | **Last Update**: 2025-08-21
