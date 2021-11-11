import Q
import numpy as np


def test(agent):

	trials, total = 100, 0
	failed, passed = 0, 0

	for episode in range(trials):

		observation = agent.environment.reset()
		state = agent.sampler(observation)
		done = False

		while not done:

			agent.environment.render()
			action = np.argmax(agent.q[state])
			observation, reward, done, information = agent.environment.step(action)
			future_state = agent.sampler(observation)
			state = future_state

		if (observation[0] >= 0.5): passed, condition = passed + 1, "Passed"
		elif (observation[0] < 0.5): failed, condition = failed + 1, "Failed"
		print("Episode:", episode + 1)
		print("Completion status:", condition, "\n")
		
	agent.environment.close()
	if (passed == 1): print(f"The agent sucessfully passed {passed} trial and failed {failed} attempts.\n")
	elif (failed == 1): print(f"The agent sucessfully passed {passed} trials and failed {failed} attempt.\n")
	else: print(f"The agent sucessfully passed {passed} trials and failed {failed} attempts.\n")


if __name__ == "__main__":

	learning_rate = 0.1
	discount_rate = 0.98
	exploration_rate = 0.001
	decay_rate = 0.01
	epochs = int(100e3)
	sections = 20
	maximum_steps = 200
	indicator = 0
	system = "MountainCar-v0"
	agent = Q.QAgent(system, sections, maximum_steps, epochs, learning_rate, discount_rate, exploration_rate, decay_rate, indicator)
	agent.learn()
	agent.plot()
	test(agent)
