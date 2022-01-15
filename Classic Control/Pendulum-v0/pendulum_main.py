import gym


env = gym.make('Pendulum-v0')
episodes_total = 5

for episode in range(episodes_total):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()  # TODO: this is of type box. Need to input appropriate value.
        observation, reward, done, info = env.step(action)
        score += reward
    print(f'Episode {episode} score: {score}')
env.close()

print(env.action_space)  # Box([-2.], [2.], (1,), float32)
