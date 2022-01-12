import gym


env = gym.make('CartPole-v1')
episodes_total = 5

for episode in range(episodes_total):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        observation, reward, done, info = env.step(action)
        score += reward
    print(f'Episode {episode} score: {score}')
env.close()

print(env.action_space)
