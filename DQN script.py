
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import time
import gymnasium as gym
from gymnasium import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MazeEnv(gym.Env):
    def __init__(self, maze_size=10, fixed_maze=None, seed=42):
        super(MazeEnv, self).__init__()

        self.maze_size = maze_size
        self.observation_space = spaces.Box(low=0, high=1, shape=(maze_size, maze_size, 3), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.seed_value = seed
        random.seed(seed)
        np.random.seed(seed)

        if fixed_maze is not None:
            self.maze = fixed_maze
        else:
            self.maze = self._generate_maze()

        self.original_maze = self.maze.copy()

        self.start_pos = (1, 1)
        self.goal_pos = (maze_size-2, maze_size-2)

        self.current_pos = self.start_pos

        self.max_steps = maze_size * maze_size * 2
        self.current_step = 0

    def _generate_maze(self):
        maze = np.ones((self.maze_size, self.maze_size), dtype=np.uint8)

        def carve_paths(x, y):
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.maze_size and 0 <= ny < self.maze_size and maze[ny, nx] == 1:
                    maze[ny, nx] = 0
                    maze[y + dy // 2, x + dx // 2] = 0
                    carve_paths(nx, ny)

        start_x, start_y = 1, 1
        maze[start_y, start_x] = 0
        carve_paths(start_x, start_y)

        maze[1, 1] = 0
        maze[self.maze_size-2, self.maze_size-2] = 0

        return maze

    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset(seed=self.seed_value)

        # Reset position but keep the same maze
        self.current_pos = self.start_pos
        self.current_step = 0
        self.maze = self.original_maze.copy()

        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        obs = np.zeros((self.maze_size, self.maze_size, 3), dtype=np.float32)

        obs[:, :, 0] = self.maze

        y, x = self.current_pos
        obs[y, x, 1] = 1
        y, x = self.goal_pos
        obs[y, x, 2] = 1

        return obs

    def step(self, action):
        self.current_step += 1

        y, x = self.current_pos
        if action == 0:
            new_pos = (y-1, x)
        elif action == 1:
            new_pos = (y, x+1)
        elif action == 2:
            new_pos = (y+1, x)
        elif action == 3:
            new_pos = (y, x-1)
        new_y, new_x = new_pos
        if (0 <= new_y < self.maze_size and 0 <= new_x < self.maze_size and
            self.maze[new_y, new_x] == 0):
            self.current_pos = new_pos

        done = self.current_pos == self.goal_pos

        if done:
            reward = 10.0
        else:
            y, x = self.current_pos
            goal_y, goal_x = self.goal_pos
            current_dist = abs(y - goal_y) + abs(x - goal_x)
            reward = -0.1
            y_old, x_old = self.current_pos
            previous_dist = abs(y_old - goal_y) + abs(x_old - goal_x)
            if current_dist < previous_dist:
                reward += 0.2


        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = {}

        return observation, reward, done, truncated, info

    def render(self):

        rgb_maze = np.zeros((self.maze_size, self.maze_size, 3), dtype=np.uint8)


        wall_indices = np.where(self.maze == 1)
        rgb_maze[wall_indices] = [0, 0, 0]


        path_indices = np.where(self.maze == 0)
        rgb_maze[path_indices] = [255, 255, 255]


        y, x = self.start_pos
        rgb_maze[y, x] = [0, 0, 255]

        y, x = self.goal_pos
        rgb_maze[y, x] = [0, 255, 0]

        y, x = self.current_pos
        rgb_maze[y, x] = [255, 0, 0]

        plt.figure(figsize=(6, 6))
        plt.imshow(rgb_maze)
        plt.title(f"Step: {self.current_step}")
        plt.axis('off')
        plt.pause(0.01)
        plt.draw()

class MazeDQN(nn.Module):
    def __init__(self, input_channels=3, maze_size=10, num_actions=4):
        super(MazeDQN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        dummy_input = torch.zeros(1, input_channels, maze_size, maze_size)

        dummy_output = self.conv3(self.conv2(self.conv1(dummy_input)))

        conv_output_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):

        batch_size = x.size(0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(batch_size, -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class DQNAgent:
    def __init__(self, state_shape, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.95, memory_size=10000,
                 batch_size=64, maze_size=10):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.maze_size = maze_size

        self.policy_net = MazeDQN(input_channels=state_shape[2], maze_size=maze_size, num_actions=action_size).to(device)
        self.target_net = MazeDQN(input_channels=state_shape[2], maze_size=maze_size, num_actions=action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(device)  
        with torch.no_grad():
            act_values = self.policy_net(state_tensor)
        return torch.argmax(act_values).item()

    def replay(self, target_update_freq=10):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        states_tensor = torch.FloatTensor(states).permute(0, 3, 1, 2).to(device)  # BCHW format
        next_states_tensor = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        dones_tensor = torch.FloatTensor(dones).to(device)

        q_values = self.policy_net(states_tensor).gather(1, actions_tensor)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor).max(1)[0]


        target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

        loss = self.loss_fn(q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_maze_agent(maze_size=10, episodes=50, render=False, seed=42):
  
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    env = MazeEnv(maze_size=maze_size, seed=seed)
    fixed_maze = env.maze.copy()

    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = DQNAgent(state_shape, action_size, maze_size=maze_size)
    agent.epsilon_decay = 0.94

    scores = []
    steps_per_episode = []
    solutions_found = 0
    losses = []
    print("Training on the following maze:")
    plt.figure(figsize=(8, 8))
    plt.imshow(fixed_maze, cmap='binary')
    plt.title("Fixed Maze for Training")
    start_y, start_x = env.start_pos
    goal_y, goal_x = env.goal_pos
    plt.plot(start_x, start_y, 'bo', markersize=10, label='Start')
    plt.plot(goal_x, goal_y, 'go', markersize=10, label='Goal')
    plt.legend()
    plt.savefig("fixed_maze.png")
    plt.show()
    print("Starting training for 50 episodes...")
    start_time = time.time()

    for episode in range(episodes):
        state, _ = env.reset()
        state = np.array(state)
        total_reward = 0
        done = False
        truncated = False
        episode_losses = []

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.array(next_state)
            total_reward += reward

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if render and (episode % 10 == 0 or episode == episodes - 1):
                env.render()

            loss = agent.replay()
            if loss != 0:
                episode_losses.append(loss)

        if episode % 5 == 0:
            agent.update_target_net()

        scores.append(total_reward)
        steps_per_episode.append(env.current_step)

        if done: 
            solutions_found += 1

        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(avg_loss)

        avg_score = np.mean(scores[-min(10, len(scores)):])
        avg_steps = np.mean(steps_per_episode[-min(10, len(steps_per_episode)):])
        success_rate = solutions_found / (episode + 1)
        print(f"Episode: {episode+1}/{episodes}, Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}, " +
              f"Epsilon: {agent.epsilon:.4f}, Steps: {env.current_step}, " +
              f"Success: {'Yes' if done else 'No'}, Success Rate: {success_rate:.2f}, " +
              f"Avg Loss: {avg_loss:.4f}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final success rate: {solutions_found/episodes:.2f}")

    # Save the model
    agent.save(f"maze_dqn_{maze_size}x{maze_size}.pt")

    # Plot training results
    plt.figure(figsize=(15, 10))

    plt.subplot(221)
    plt.plot(scores)
    plt.title('Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    plt.subplot(222)
    plt.plot(steps_per_episode)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.subplot(223)
    window_size = min(10, len(scores))
    success_rates = []
    for i in range(len(scores) - window_size + 1):
        success_rate = sum(1 for s in range(i, i + window_size) if scores[s] > 0) / window_size
        success_rates.append(success_rate)

    plt.plot(range(window_size-1, len(scores)), success_rates)
    plt.title('Success Rate (Rolling Window)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')

    plt.subplot(224)
    plt.plot(losses)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(f"maze_training_{maze_size}x{maze_size}.png")
    plt.show()

    return agent, env, fixed_maze

def visualize_agent_path(agent, env, fixed_maze, max_steps=200):
   
    env = MazeEnv(maze_size=env.maze_size, fixed_maze=fixed_maze)
    state, _ = env.reset()
    state = np.array(state)
    done = False
    truncated = False
    steps = 0

    plt.figure(figsize=(10, 10))

    path = [env.current_pos]
    actions_taken = []

    while not (done or truncated) and steps < max_steps:
       
        agent.epsilon = 0
        action = agent.select_action(state)
        actions_taken.append(action)

        next_state, reward, done, truncated, _ = env.step(action)
        state = np.array(next_state)

        path.append(env.current_pos)
        steps += 1
    rgb_maze = np.zeros((env.maze_size, env.maze_size, 3), dtype=np.uint8)

    wall_indices = np.where(env.maze == 1)
    rgb_maze[wall_indices] = [50, 50, 50]

    path_indices = np.where(env.maze == 0)
    rgb_maze[path_indices] = [220, 220, 220]

    y, x = env.start_pos
    rgb_maze[y, x] = [0, 0, 255]

    y, x = env.goal_pos
    rgb_maze[y, x] = [0, 255, 0]
    for i, (y, x) in enumerate(path):
        intensity = min(255, int(255 * i / len(path)))
        rgb_maze[y, x] = [255, 255 - intensity, 0]

    plt.imshow(rgb_maze)

    start_y, start_x = env.start_pos
    goal_y, goal_x = env.goal_pos
    plt.scatter(start_x, start_y, c='blue', s=100, marker='o')
    plt.scatter(goal_x, goal_y, c='green', s=100, marker='*')

    action_to_arrow = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    for i in range(len(path)-1):
        y, x = path[i]
        plt.text(x, y, action_to_arrow[actions_taken[i]],
                 horizontalalignment='center', verticalalignment='center',
                 color='black', fontweight='bold')

    title = "Agent Path"
    if done:
        title += f" - Goal reached in {steps} steps!"
    else:
        title += f" - Failed to reach goal ({steps} steps)"

    plt.title(title)
    plt.axis('off')
    plt.savefig("agent_path.png")
    plt.show()

    return done, steps

def main():
    MAZE_SIZE = 15
    EPISODES = 100
    SEED = 42

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print(f"Training agent on {MAZE_SIZE}x{MAZE_SIZE} maze for {EPISODES} episodes...")
    agent, env, fixed_maze = train_maze_agent(maze_size=MAZE_SIZE, episodes=EPISODES, render=True, seed=SEED)

    print("\nVisualizing agent's path...")
    success, steps = visualize_agent_path(agent, env, fixed_maze)

    if success:
        print(f"Success! Agent reached the goal in {steps} steps.")
    else:
        print(f"Failed to reach the goal in {steps} steps.")

if __name__ == "__main__":
    main()
