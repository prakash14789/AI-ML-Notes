# 86. RL in Practice - Suman - 30 May 2026

# Reinforcement Learning in Practice

## 1. What You'll Learn in This Section

In this lesson, you'll learn to:

- Distinguish reinforcement learning from supervised and unsupervised learning and explain what makes it unique
- Identify the core components of an RL system — agent, environment, state, action, reward, and policy — and describe how they interact
- Apply the Q-learning algorithm to a grid-world problem using the Gymnasium FrozenLake environment
- Explain how PPO keeps policy updates stable and how RLHF uses human feedback to align language models

## 2. Detailed Explanation

### The three paradigms of machine learning

Machine learning is typically divided into three paradigms, each defined by the kind of signal the algorithm uses to learn.

**Supervised learning** works with a labelled dataset — every input example has a corresponding output label Y. Algorithms like linear regression, logistic regression, random forest, and XGBoost all fall here. The goal is to learn the mapping from inputs to Y, and success is measured by minimising a loss or cost function.

**Unsupervised learning** has no label Y. The algorithm looks for structure within the data itself — grouping similar items (K-means clustering, DBSCAN) or compressing it into fewer dimensions (PCA, factor analysis). Success here is measured by metrics such as silhouette score, which captures how tight and well-separated the clusters are.

**Reinforcement learning** is the third paradigm, and it works differently from both. The agent is not given labelled input-output pairs at all. Instead, it learns by interacting with an environment and receiving feedback signals — rewards when it does well, penalties when it does not. All three paradigms share one underlying goal: minimise error (or equivalently, maximise accuracy). In RL, this means maximising cumulative reward.

```
Supervised Learning(labelled data → minimise loss)Unsupervised Learning(no labels → minimise cluster distance)Reinforcement Learning(no labels → maximise cumulative reward)Machine Learning Paradigms
```

### What reinforcement learning is — and the lunchbox analogy

**Reinforcement learning** is a type of machine learning where an agent learns to make decisions by interacting with an environment in order to maximise cumulative reward.

The clearest way to build intuition is through the example of a child learning to finish their school lunchbox:

- The parent acts as the **environment** — they observe outcomes and deliver feedback, but never explicitly tell the child what to do. The child must infer the goal.
- The child acts as the **agent** — it takes actions (eating more or less food).
- If the lunchbox comes home unfinished, the parent scolds and withholds candy — a **penalty**.
- If the lunchbox is finished, the parent praises the child and offers chocolate — a **reward**.
- Over repeated episodes, the child's behaviour adapts to maximise reward and avoid penalty.

The machine analogue follows exactly the same logic. The key clarification: the agent is never told what to do — it must discover the correct behaviour through trial and error, guided purely by the reward signal.

One important difference from humans: machines have no feelings or intrinsic motivation. The agent is simply programmed to maximise a mathematical quantity — reward points. It has no preference for what those reward points represent; it is purely driven by the objective of accumulating the highest possible total value.

### Key terminology

Before going further, here is the vocabulary you will encounter throughout RL. Every term has a precise meaning.

Term | Definition
Agent | The learner or decision-maker that interacts with the environment
Environment | The world or system the agent acts within
State (S_t) | The representation of the current situation at time step t
Action (α_t) | A choice made by the agent from the set of possible actions at time step t
Reward (R) | Immediate feedback from the environment — positive encourages the action, negative discourages it
Policy (π) | The strategy the agent uses to decide which action to take in each state
Value function (V^π) | Maps each state to the expected cumulative reward when following policy π
Action value function (Q) | Expected cumulative reward from a given state–action pair (also called the Q-function)
Discount factor (γ) | Balances immediate vs future rewards — close to 1 prioritises long-term, close to 0 prioritises immediate
Episode | One complete run from the initial state to a terminal condition
Transition (T) | The mechanism by which action A in state S produces a new state S'

The notation summary: S_t is state at time t, S_{t+1} is the next state, α_t is action at time t, π is policy, γ is the discount factor, and Q(S, A) is the action value function.

### The RL interaction loop — how the agent learns

At each time step, the agent and environment follow a six-step cycle:

1. The agent observes current state S_t.
2. It selects action A_t according to its current policy π.
3. The environment transitions to new state S_{t+1}.
4. The environment returns reward R.
5. The agent updates its policy or value estimates based on the feedback.
6. Repeat from step 1.

This loop continues until the objective is met: find a policy π that maximises the expected cumulative (discounted) reward over all time steps.

Three characteristics make RL distinct from the other paradigms:

- **No labelled data required** — the agent generates its own experience through interaction.
- **Delayed rewards** — a good action taken now may not pay off until many steps later.
- **Exploration vs exploitation** — the agent must balance trying new actions (exploration) against repeating actions already known to be good (exploitation).

RL shows up in practice in gameplay (board games, video games), robotics, and recommendation systems. A concrete example: when a user clicks "I don't like this" on a Netflix suggestion, the system penalises the policy that produced that recommendation and adjusts future behaviour toward content the user will reward.

### Grid-world navigation — optimal paths and traps

A simple grid world makes these concepts visual. The agent starts in one corner and must reach a goal cell. Multiple routes exist, but the agent's objective is to find the **optimal path** — the one that reaches the goal in the minimum number of steps.

Some cells are **traps** (blockades). Entering a trap ends the episode unsuccessfully. Unnecessary detours also cost the agent reward points: each extra step means fewer reward points per episode compared to the shortest path. This generalises to any optimisation problem where the state space has millions of possible paths.

### The Bellman equation

The **Bellman equation** is the mathematical foundation of dynamic programming and reinforcement learning. It expresses the value of being in a state as the sum of two parts: the immediate reward from that state, plus the discounted expected value of all future states.

The formal expression for the state value function under policy π is:

```
V^π(S_t) = E_π [R_{t+1} + γ · V^π(S_{t+1}) | S_t]

```

Where:

- `E_π[·]` — expectation over actions chosen by policy π
- `R_{t+1}` — immediate reward after taking the action
- `γ` — discount factor
- `V^π(S_{t+1})` — discounted value of the next state

The Bellman equation relies on the **Markov property**: the future depends only on the present state, not on the history of how the agent arrived there. This is what makes RL mathematically tractable — because only the current state matters, the equation applies step by step without needing to track history.

### Markov Decision Process (MDP)

Before training an agent, the world must be formally defined as a **Markov Decision Process (MDP)**. The MDP specifies all the moving parts:

Symbol | Component | Meaning
S | State space | All possible states the agent can be in
A | Action space | All possible actions the agent can take
R | Reward | The feedback signal
S' (S-prime) | Next state | The state reached after taking an action
T | Transition | The rule governing how actions change states

The Markov property sits at the centre of this definition: the transition to the next state depends only on the current state and chosen action — earlier history is irrelevant. Gymnasium environments are explicitly built around this structure.

### Q-learning — building the Q-table

**Q-learning** is the foundational RL algorithm for discrete state–action spaces. It builds a **Q-table** — a lookup table where each entry Q(S, A) stores the expected total future reward for taking action A in state S.

**Initialisation:** The Q-table starts as all zeros, because the agent begins with no knowledge.

**Update rule (Bellman update / TD target):**

```
Q(S, A) ← Q(S, A) + α · [TD_target − Q(S, A)]

TD_target = R + γ · max_{A'} Q(S', A')

```

Where α (alpha) is the learning rate, γ (gamma) is the discount factor, R is the reward received, S' is the next state, and `max Q(S', A')` is the best Q-value achievable from the next state. The update moves the current Q-estimate toward what it "should have been" (the TD target), scaled by the learning rate.

**Exploration vs exploitation — epsilon-greedy strategy:** The agent uses a parameter ε (epsilon):

- With probability ε, the agent picks a random action (exploration — it does not yet know which direction to go).
- With probability 1−ε, the agent picks the action with the highest current Q-value (exploitation — using already-learned knowledge).

Epsilon is a hyperparameter set by the practitioner. The full set of Q-learning parameters is: alpha, gamma, epsilon, epsilon_minimum, and number of episodes.

### Gymnasium and the FrozenLake environment

**Gymnasium** is a Python library of pre-built environments driven by Markov Decision Processes. It is the standard toolkit for practising RL algorithms in code. It was formerly known as OpenAI Gym. The required libraries for the implementation are `matplotlib`, `ipywidgets`, `gymnasium`, and `stable-baselines3`.

```python
import gymnasium as gym
import matplotlib.pyplot as plt
import ipywidgets

```

**FrozenLake-v1** is a 4×4 grid environment provided by Gymnasium. The grid uses four cell symbols:

Symbol | Meaning
S | Start position
F | Frozen (safe) tile
H | Hole (trap)
G | Goal

The objective is to navigate from S to G without stepping into any H hole. FrozenLake resembles the board game Snakes and Ladders: holes correspond to snakes (traps that end the episode), reaching G corresponds to winning. The key difference — in Snakes and Ladders moves are dice-driven; in FrozenLake the agent deliberately chooses each action.

The environment is created with:

```python
env = gym.make("FrozenLake-v1")

```

The state space is 16 states (positions 0–15 on the 4×4 grid). The action space is 4 discrete actions — left, down, right, up. Hole positions are at indices 5, 7, 11, and 12. An episode ends when the agent reaches G (success) or steps into an H (failure, reward = 0).

### Q-learning implementation for FrozenLake

The Q-table is initialised as a matrix of zeros with shape (n_states, n_actions) = (16, 4):

```python
import numpy as np
Q = np.zeros((n_states, n_actions))

```

The training loop runs for a specified number of episodes. At each step: the agent selects an action using epsilon-greedy, the environment returns the next state and reward, and the Q-table is updated using the Bellman update formula.

Several helper functions support the implementation:

- A function to remove noise from reward curves (smoothing)
- A function to draw the grid
- A function to draw the learned policy (policy arrows overlaid on the grid showing the best action per cell)
- A function to plot Q-values
- A `RewardLogger` class that records total reward per completed episode during training
- A benchmarking function

**Reading the output graphs:**

- **Episode vs success rate graph** — early episodes show a low success rate because the agent is randomly exploring. As learning progresses the success rate climbs as the agent identifies the safe paths.
- **Q-value heatmap per action** — for each of the four actions (left, down, right, up), the table shows which cells yield high expected reward and which yield low reward. For example, taking "down" in cell 12 produces a high Q-value (moves toward G), while "down" at cell 15 produces a low Q-value.
- **Policy arrows** — overlaid arrows show the best action the agent has learned for each safe cell.

**Watch out for:** The Q-table only works well in practice when the state space is small and discrete. FrozenLake has 16 states, so a (16, 4) table is manageable. For environments with continuous state variables — like CartPole — a table becomes impractical and more powerful approaches are needed.

### CartPole — a continuous-state environment

**CartPole** is a Gymnasium environment that illustrates continuous-state RL. The objective is to apply left or right forces to a cart to keep a pole balanced upright for as long as possible.

The state at each step is described by four observation variables: pole angle, pole angular velocity, cart position, and cart velocity. The action space has two choices — 0 (push left) or 1 (push right). The reward is +1 for every time step the pole remains upright. An episode terminates when the cart position exceeds ±2.4 or the pole angle exceeds ±12 degrees. Penalty types include angle penalty, position penalty, action-change penalty, velocity penalty, and early-termination penalty.

### PPO — keeping policy updates stable

**Proximal Policy Optimization (PPO)** is a policy optimisation algorithm introduced by OpenAI. It addresses a key challenge that appears when updating the policy using gradient-based methods: large updates can destabilise training.

The core goal of PPO is to improve the policy while keeping policy updates small and stable. The intuition: start with an old policy and update toward a better one, but constrain how far the new policy can deviate. Drastic changes move the agent into unfamiliar territory with unpredictable behaviour — so PPO uses a **clipped surrogate objective function**. The objective is computed as the minimum of the actual policy improvement and a clipped version of it. This prevents any single update from being too large. The objective is taken as an expectation over trajectories.

The `stable-baselines3` library provides a ready-to-use PPO implementation that can be applied to any Gymnasium environment — making it the most practical entry point for applying PPO in real projects.

### Reinforcement Learning from Human Feedback (RLHF)

**RLHF** applies RL concepts to large language model (LLM) alignment. The agent is a pre-trained language model; human evaluators serve as the environment that provides reward signals. The pipeline has five steps:

1. **Start with a pre-trained (supervised fine-tuned) model** — an LLM already trained on text data.
2. **Generate multiple responses** — the model produces several candidate responses to a prompt (for example, labelled Response A, B, C).
3. **Collect human preference rankings** — human evaluators rank the responses by quality. A pairwise interface (show two options; pick one) works better than asking for a full ordering. Elimination is the easiest decision, so users are more likely to complete it reliably.
4. **Train a reward model** — a reward model is trained to predict human preference scores from the collected rankings. It acts as a policy: given a new prompt, it predicts which response style the user is most likely to prefer.
5. **RL fine-tuning** — the language model is fine-tuned using RL to maximise the reward predicted by the reward model. This adjusts how the model formats and presents output — not the underlying weights for factual knowledge.

An important distinction: RLHF does not retrain the model's weights in real time. What it updates is the model's style, formatting, and output strategy — not its factual knowledge.

Why use RLHF in practice? Human feedback captures complex preferences — helpfulness, safety, and truthfulness from the end-user's perspective — that are difficult to encode as a loss function. A technically correct response is not useful if it is incomprehensible to the target audience.

When a user clicks thumbs up or thumbs down on a ChatGPT response, they are providing an RL signal. The system penalises the policy that produced a disliked response and rewards the policy that produced a liked one. Full RLHF is computationally expensive and typically applied to large-scale production systems; RL fine-tuning is more feasible for smaller or medium-sized language models.

## 3. Key Takeaways

- Reinforcement learning is the third ML paradigm — unlike supervised or unsupervised learning, the agent learns entirely through interaction, receiving rewards and penalties rather than labelled data. This makes RL the natural fit for sequential decision-making tasks that appear throughout industry practice.
- The core loop is: observe state → select action → receive reward → update policy — repeat until cumulative reward is maximised.
- The Bellman equation formalises state value as immediate reward plus discounted future value; the Markov property makes this decomposition tractable by requiring only the current state.
- Q-learning builds a Q-table of expected future rewards for every state–action pair; epsilon-greedy balances exploration (random actions) against exploitation (best-known actions). In practice, Gymnasium provides ready-made environments — FrozenLake and CartPole — to test these ideas directly.
- RLHF brings the same agent–reward loop to LLM alignment — human evaluators act as the environment, a reward model captures their preferences, and RL fine-tuning adjusts the model's output style toward what humans find helpful, safe, and truthful.

**Mental model:** Think of reinforcement learning as training an agent the way you would train a puppy — you never explain the rules in words; you simply reward the behaviours you want and discourage the ones you don't, and the agent figures out the rest through repeated experience.

            .markdown-preview table, 
            .markdown-preview th, 
            .markdown-preview td {
              background-color: white !important;
              color: black !important;
            }
            .markdown-preview pre, 
            .markdown-preview code {
              background-color: inherit !important;
              color: inherit !important;
              box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }