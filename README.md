# gymnasium

Project to explore gymnasium environments.

## Install

```bash
git clone https://github.com/SonDePoisson/gymnasium.git
```

## Requirements

- `Gymnasium` : Get all environments to work in,
- `Stable_Baselines3` : (SB3) to ease Gymnasium usage,
- `TensorBoard` : Plot all data in TensorBoard.

```bash
pip install -r requirements.txt
```

`SB3` might not install successfully through requirements file if your using ZSH. In this case, please run before :

```bash
pip install 'stable-baselines3[extra]'
```

And then run again the first install.

## Project Structure

- `train/` : Training scripts for various environments.
- `run/` : Scripts to run and visualize trained agents.
- `models/` : Saved models after training.
- `logs/` : TensorBoard logs and training statistics.

## Usage

The project is divided in 2 sections (Training and Running).
For each section, make sure to run the scripts from the root of the project to create `models/` and `logs/` in the root folder.

### Training

To train an agent, run one of the training scripts in the `train/` directory.  
For example :

```bash
python train/train_MJCReacherParallel.py
```

### Running

To run a trained agent and visualize its behavior, use the corresponding script in the `run/` directory.  
For example :

```bash
python run/run_MJCReacher.py
```

### Monitoring with TensorBoard

You can monitor training progress and visualize metrics using TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open the provided URL in your browser.

## Sources

[PythonProgramming SB3 Tuto](https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/)

[Gymnasium Doc](https://gymnasium.farama.org)

[SB3 Doc](https://stable-baselines3.readthedocs.io/en/master/index.html)

## Author

Project by Clément Poisson.
