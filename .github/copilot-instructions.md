# SecureML Ops — Project Guidelines

## What This Is

Adversarial robustness testing platform — "VirusTotal but for AI models."
User uploads a trained ML model, selects attacks, platform attacks the model and returns a Robustness Score (0–100) with defense recommendations.

## Tech Stack

- **Backend**: Python 3.10, FastAPI, Celery, Redis, ART (adversarial-robustness-toolbox by IBM), PyTorch, TorchVision, MLflow, SQLite, ReportLab, Matplotlib
- **Frontend**: React 18, Vite, Tailwind CSS, Recharts, Lucide React, Axios
  - Use plain `useEffect` for polling — **NO** React Query
  - Native HTML drag-and-drop — **NO** react-dropzone
  - **NO** framer-motion

## Model Format Priority

- **Primary**: PyTorch (`.pt` / `.pth`) — all new code targets PyTorch first.
- **Secondary**: ONNX (`.onnx`) and TensorFlow/Keras (`.h5`) — added after core PyTorch path works.

## Datasets

- **Phase 1**: MNIST only (torchvision built-in).
- **Phase 2+**: CIFAR-10. No custom or user-uploaded datasets.

## Model Selection

- Never hard-code a specific model architecture. Respect user's model picker.

## Code Style

- Python 3.10+, type hints on public APIs.
- Every file must be complete and immediately runnable — no `# TODO` placeholders.
- Include proper error handling (`try/except` backend, `try/catch` frontend).
- Async endpoints with FastAPI; sync-heavy ML work dispatched to Celery workers.

## Architecture

```
backend/
├── main.py, worker.py
├── routers/ (models, attacks, defenses, reports)
├── attacks/ (fgsm, pgd, cw, transfer, hopskipjump, square)
├── defenses/ (adversarial_training, preprocessing, randomized_smoothing)
├── utils/ (scorer, visualizer, tracker, defense_advisor)
└── storage/ (models/, reports/)
frontend/src/
├── pages/ (Upload, Configure, Results, History)
└── components/ (ScoreGauge, AttackCard, DefensePanel, PerturbationView)
```

## The 6 Attacks

| # | Attack | ART Class | Type | Notes |
|---|--------|-----------|------|-------|
| 1 | FGSM | `FastGradientMethod` | white-box | single step, eps param |
| 2 | PGD | `ProjectedGradientDescent` | white-box | iterative, 40 iterations |
| 3 | C&W | `CarliniL2Method` | white-box | optimization-based, 100-sample batch (slow) |
| 4 | Transfer | manual build | grey-box | surrogate CNN → PGD on surrogate → test on target |
| 5 | HopSkipJump | `HopSkipJump` | black-box | decision boundary walk, needs only labels |
| 6 | Square | `SquareAttack` | black-box | random patches, needs only confidence scores |

Every attack returns:
```python
{"attack": str, "type": str, "clean_accuracy": float,
 "adv_accuracy": float, "epsilon": float, "x_adv": np.ndarray}
```

## Robustness Scoring

Weights: FGSM 0.10, PGD 0.20, C&W 0.20, Transfer 0.20, HopSkipJump 0.15, Square 0.15.
Score = weighted sum of `(adv_accuracy / clean_accuracy)` × 100.

## Build & Test

```bash
pip install -r backend/requirements.txt
pytest backend/tests/ -v
cd frontend && npm ci && npm test
```

## Git Commits

`feat(engine):` attack/scoring · `feat(api):` routes · `feat(ui):` React · `feat(defense):` defenses · `chore:` setup · `fix:` bugs

## Build Order

1. Attack Engine: fgsm → pgd → cw → transfer → hopskipjump → square → scorer → visualizer
2. Defenses: adversarial_training → preprocessing → randomized_smoothing → defense_advisor
3. Backend API: main.py → routers → worker + attacks router → tracker
4. Frontend: Vite setup → Upload → Configure → Results → DefensePanel
5. Ship: docker-compose → Vercel config → README
