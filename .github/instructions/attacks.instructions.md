---
description: "Use when creating, editing, or reviewing adversarial attack modules under backend/attacks/. Covers FGSM, PGD, C&W, Transfer, HopSkipJump, Square, and any new attack type."
applyTo: "backend/attacks/**"
---
# Attack Module Guidelines

## Structure

Each attack lives in its own file under `backend/attacks/` and inherits from `BaseAttack` in `backend/attacks/base.py`.

Required public interface:

```python
class <Attack>(BaseAttack):
    def run(self, model: nn.Module, data: tuple[np.ndarray, np.ndarray], **kwargs) -> dict:
        ...
```

## ART Integration

- White-box attacks wrap ART evasion classes (`FastGradientMethod`, `ProjectedGradientDescent`, `CarliniL2Method`).
- Black-box attacks wrap `HopSkipJump`, `SquareAttack`.
- Transfer attack is built manually (surrogate CNN → PGD on surrogate → test on target).
- Wrap the PyTorch model in `art.estimators.classification.PyTorchClassifier` before passing to ART.

## Return Format

Every `run()` must return this dict:

```python
{
    "attack": "ATTACK_NAME",
    "type": "white-box" | "grey-box" | "black-box",
    "clean_accuracy": float,
    "adv_accuracy": float,
    "epsilon": float,
    "x_adv": np.ndarray,
}
```

## Implementation Rules

1. **PyTorch first** — models arrive as `nn.Module`, wrap in ART's `PyTorchClassifier`.
2. **Data as numpy** — `run()` receives `(x, y)` as numpy arrays, shape `(N, C, H, W)` in `[0, 1]`.
3. **Epsilon** — accept as constructor param with sensible default; pass to ART attack.
4. **Batch support** — process entire batch in one ART `generate()` call.
5. **No placeholders** — every file must be complete and runnable, no `# TODO`.
6. **Error handling** — wrap attack execution in try/except, return clean error info on failure.

## Testing

- Unit tests in `backend/tests/attacks/test_<name>.py`.
- Phase 1: MNIST. Phase 2: CIFAR-10.
- Assert: (a) adversarial examples differ from originals, (b) perturbation within epsilon, (c) output shape matches input, (d) clean_accuracy and adv_accuracy are valid floats.
