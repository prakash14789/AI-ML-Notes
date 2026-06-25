# 90. Reinforcement Learning & Capstone - Suman - 6 Jun 2026

# Reinforcement Learning and Capstone

## 1. What You'll Learn in This Section

In this lesson, you'll learn to:

- Model sequential decision problems as Markov Decision Processes (MDPs)
- Explain value functions, policies, rewards, and the Bellman equation
- Implement tabular Q-learning and reason about exploration versus exploitation
- Distinguish value-based methods from policy-gradient methods
- Describe how Reinforcement Learning from Human Feedback (RLHF) works
- Evaluate reinforcement-learning systems through governance, safety, and demonstration design

## 2. Detailed Explanation

### From supervised learning to sequential decisions

Supervised learning maps an input to a known target. Reinforcement learning (RL) is different: an **agent** repeatedly acts inside an **environment**, observes the result, and receives a reward. A good action may create a benefit only several steps later, so the agent must learn long-term consequences rather than copy labelled answers.

```
action a_tstate s_(t+1), reward r_(t+1)AgentEnvironment
```

Examples include warehouse routing, game playing, recommendation sequencing, robot control, and aligning language models with human preferences.

### Markov Decision Processes

An MDP is commonly represented by the tuple:

```
(S, A, P, R, gamma)

```

Component | Meaning
S | Set of possible states
A | Set of available actions
P(s' | s, a) | Probability of reaching states'after actionain states
R(s, a, s') | Immediate reward for the transition
gamma | Discount factor between 0 and 1

The **Markov property** says that the current state contains all information needed to predict the next state. The complete history should not be required once the current state is known.

A **policy**, written as `pi(a|s)`, defines the agent's behaviour. It may choose one action deterministically or assign probabilities to several actions.

### Return, discounting, and value functions

The return from time `t` is the discounted sum of future rewards:

```
G_t = r_(t+1) + gamma*r_(t+2) + gamma^2*r_(t+3) + ...

```

A lower `gamma` makes the agent care more about immediate rewards. A value close to 1 gives more weight to long-term outcomes.

Two value functions are central:

- `V^pi(s)` estimates the expected return from state `s` while following policy `pi`.
- `Q^pi(s, a)` estimates the expected return after taking action `a` in state `s` and then following `pi`.

The Bellman optimality equation for action values is:

```
Q*(s, a) = E[r + gamma * max_a' Q*(s', a')]

```

It breaks a long decision problem into an immediate reward plus the best estimated future value.

### Q-learning

Q-learning is an **off-policy, model-free** algorithm. It does not need the transition probabilities, and it learns the value of the greedy target policy even when the behaviour policy explores.

The update rule is:

```
Q(s, a) <- Q(s, a) + alpha * [r + gamma*max_a' Q(s', a') - Q(s, a)]

```

The expression inside brackets is the **temporal-difference error**. `alpha` is the learning rate.

```python
import numpy as np

q_table = np.zeros((num_states, num_actions))

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))

        next_state, reward, done = env.step(action)
        td_target = reward + gamma * np.max(q_table[next_state]) * (not done)
        q_table[state, action] += alpha * (td_target - q_table[state, action])
        state = next_state

```

**Watch out for:** terminal states have no future return. Multiplying the future-value term by `(not done)` prevents value from leaking beyond the end of an episode.

### Exploration versus exploitation

An agent must balance:

- **Exploitation**: choose the action currently believed to be best.
- **Exploration**: try less-certain actions that may produce better outcomes.

An epsilon-greedy policy chooses a random action with probability `epsilon` and the greedy action otherwise. Training often begins with a high epsilon and gradually decays it.

Too little exploration can trap the agent in a poor policy. Too much exploration prevents stable high-reward behaviour. Evaluation should normally use the learned policy without exploratory actions.

### Policy gradients

Q-learning selects actions by comparing learned values. A policy-gradient method instead represents the policy directly as `pi_theta(a|s)` and adjusts parameters `theta` to increase expected return.

The REINFORCE estimator has the form:

```
gradient J(theta) ~= sum_t gradient log pi_theta(a_t|s_t) * G_t

```

Actions followed by high returns become more likely; actions followed by poor returns become less likely.

Policy gradients are useful for continuous action spaces and stochastic policies, but they can have high variance. A **baseline**, often a learned value function, reduces variance without changing the expected gradient. Actor-critic systems combine:

- An **actor** that selects actions
- A **critic** that estimates value and supplies a learning signal

### RLHF

Reinforcement Learning from Human Feedback applies preference information to model alignment. A simplified pipeline is:

```
Supervised fine-tuningGenerate response pairsHumans rank responsesTrain reward modelOptimise policy against rewardSafety and quality evaluation
```

1. Start with a pretrained model and supervised fine-tuning.
2. Collect human comparisons between candidate responses.
3. Train a reward model to predict those preferences.
4. Optimise the policy to obtain high predicted reward, usually while constraining it from drifting too far from the reference model.

The reward model is a proxy, not a perfect definition of human values. An optimiser may exploit weaknesses in that proxy, a failure known as **reward hacking**.

### Governance and responsible deployment

RL systems need controls beyond average reward:

Governance area | Practical control
Objective design | Review reward definitions and unintended incentives
Data provenance | Record where demonstrations and preference labels came from
Safety | Define forbidden actions, limits, and shutdown behaviour
Evaluation | Test rare, adversarial, and distribution-shift scenarios
Human oversight | Require approval for high-impact decisions
Monitoring | Track reward, policy drift, failures, and subgroup outcomes
Reproducibility | Version code, seeds, environments, data, and checkpoints

For high-impact applications, deployment should be staged: offline testing, simulation, a limited pilot, monitored rollout, and a rollback plan.

### Capstone demonstration

A strong capstone makes the decision process visible. For example, build a grid-world delivery agent:

1. Define states as grid positions plus package status.
2. Define actions as movement and pickup/drop-off operations.
3. Assign a small step penalty, a collision penalty, and a delivery reward.
4. Train a Q-learning baseline.
5. Plot episode return, success rate, and path length.
6. Demonstrate the final greedy policy on fixed test maps.
7. Document unsafe shortcuts, reward-design risks, and governance controls.

The demo should compare the trained agent with a random policy. A rising training reward alone is not enough; report evaluation performance across multiple seeds and unseen starting states.

### Common failure modes

**Reward hacking** occurs when the agent maximises the stated reward in an unintended way. Inspect behaviours, not only totals.

**Sparse rewards** provide too little learning signal. Reward shaping can help, but shaping must not change the desired final objective.

**Unstable learning** may result from a high learning rate, unscaled rewards, insufficient exploration, or function-approximation errors.

**Offline-to-online mismatch** occurs when simulations or logged data do not represent deployment conditions. Use stress tests and conservative rollout limits.

## 3. Key Takeaways

- An MDP formalises states, actions, transitions, rewards, and discounting for sequential decisions.
- Q-learning uses temporal-difference updates to learn optimal action values without a transition model.
- Policy-gradient methods optimise a parameterised policy directly and are well suited to stochastic or continuous actions.
- RLHF converts human comparisons into a learned reward signal, but that reward remains an imperfect proxy.
- Governance must cover objectives, data, safety boundaries, evaluation, monitoring, human oversight, and rollback.
- A capstone demo should show behaviour, compare baselines, evaluate across multiple conditions, and discuss failure modes.

**Mental model:** Reinforcement learning is not simply "predict the right answer." It is learning a strategy whose current actions change the future data and rewards the agent will experience.

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