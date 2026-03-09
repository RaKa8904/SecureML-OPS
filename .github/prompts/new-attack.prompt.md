---
description: "Scaffold a new adversarial attack module with boilerplate, tests, and Celery task registration."
agent: "secureml-ops"
argument-hint: "Attack name, e.g. 'DeepFool'"
---
# Scaffold new attack: {{attack_name}}

Create the following files for a new adversarial attack called **{{attack_name}}**:

1. **`backend/attacks/{{attack_name_snake}}.py`**
   - Import and subclass `BaseAttack` from `backend/attacks/base.py`.
   - Implement `__init__` with configurable hyperparameters (epsilon, iterations, etc.).
   - Implement `run(self, model, data, **kwargs) -> AttackResult`.
   - PyTorch implementation first; leave clearly-marked `# TODO: ONNX adapter` and `# TODO: TF adapter` placeholders.
   - Support both targeted and untargeted modes.

2. **`backend/tests/attacks/test_{{attack_name_snake}}.py`**
   - Use MNIST fixture for Phase 1.
   - Test perturbation bounds, output shape, and success rate > 0.

3. **Register the attack**
   - Add the new class to `backend/attacks/__init__.py` exports.
   - Add a Celery task in `backend/tasks/` that wraps the attack's `run` method.

Follow the conventions in [attacks.instructions.md](../.github/instructions/attacks.instructions.md).
