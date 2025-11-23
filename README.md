# SB3 PPO Self-Driving Car Template

This is a template project for implementing a Proximal Policy Optimization (PPO) agent using Stable-Baselines3 (SB3) for a self-driving car environment. The skeleton code is provided, and students can fill in the implementation details and experiment with hyperparameters.

## Setup Instructions

### 1. Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Once your virtual environment is activated, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

You can verify that everything is installed correctly by running:

```bash
python -c "import stable_baselines3; import gymnasium; print('Installation successful!')"
```

## Project Structure

```
sb3ppotemplate/
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── environment.py           # Custom self-driving car environment (to be implemented)
├── train.py                 # Training script (to be implemented)
├── evaluate.py              # Evaluation script (to be implemented)
└── models/                  # Directory for saved models
```

## Next Steps

1. Implement the custom self-driving car environment in `environment.py`
2. Set up the PPO training script in `train.py`
3. Create an evaluation script in `evaluate.py`
4. Experiment with hyperparameters and training configurations

## Resources

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)

## Notes

- Make sure to activate your virtual environment before running any scripts
- Models will be saved in the `models/` directory
- Adjust hyperparameters in the training script to experiment with different configurations

