import gym
from stable_baselines3 import A2C


env = gym.make('CartPole-v1')
episodes_total = 5
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Save the agent
model.save("a2c_cartpole")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = A2C.load("a2c_cartpole")


for episode in range(episodes_total):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        # action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        score += reward
    print(f'Episode {episode} score: {score}')
env.close()

print(env.action_space)
