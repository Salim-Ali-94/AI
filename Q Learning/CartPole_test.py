import Q
import numpy as np


def test(agent):

	trials, total = 100, 0
	failed, passed = 0, 0

	for episode in range(trials):

		observation = agent.environment.reset()
		state = agent.sampler(observation)
		done, score = False, 0

		while not done:

			agent.environment.render()
			action = np.argmax(agent.q[state])
			observation, reward, done, information = agent.environment.step(action)
			future_state = agent.sampler(observation)
			state = future_state
			score += reward

		if (score >= 195): passed += 1
		elif (score < 195): failed += 1
		print("Episode:", episode + 1)
		print("Score:", score, "\n")
		total += score
		
	agent.environment.close()
	if (passed == 1): print(f"The agent sucessfully passed {passed} trial and failed {failed} attempts, with an average score of {total / 100}.\n")
	elif (failed == 1): print(f"The agent sucessfully passed {passed} trials and failed {failed} attempt, with an average score of {total / 100}.\n")
	else: print(f"The agent sucessfully passed {passed} trials and failed {failed} attempts, with an average score of {total / 100}.\n")


if __name__ == "__main__":

	learning_rate = 0.1
	discount_rate = 0.98
	exploration_rate = 0.1
	decay_rate = 0.01
	epochs = int(100e3)
	sections = 20
	maximum_steps = 200
	indicator = 1
	system = "CartPole-v1"
	agent = Q.QAgent(system, sections, maximum_steps, epochs, learning_rate, discount_rate, exploration_rate, decay_rate, indicator)
	agent.learn()
	agent.plot()
	test(agent)
