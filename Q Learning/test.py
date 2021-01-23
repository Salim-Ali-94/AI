import Q


if __name__ == "__main__":

	learning_rate = 0.1
	discount_rate = 0.98
	exploration_rate = 0.1
	decay_rate = 0.01
	epochs = int(100e3)
	sections = 20
	system = "CartPole-v1"
	maximum_steps = 200
	agent = Q.QAgent(system, sections, maximum_steps, epochs, learning_rate, discount_rate, exploration_rate, decay_rate)
	agent.learn()
	agent.plot()
	agent.test()