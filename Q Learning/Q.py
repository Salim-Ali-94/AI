import gym
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"


class QAgent(object):

	Q = lambda self, reward, state, action, future_state: (1 - self.alpha)*self.q[state][action] + self.alpha*(reward + self.gamma*np.max(self.q[future_state]))

	def __init__(self, system, sections, maximum_steps, epochs, learning_rate, discount_rate, exploration_rate, decay_rate):

		self.environment = gym.make(system)
		self.environment.reset()
		self.epochs = epochs
		self.maximum_steps = maximum_steps
		self.alpha = learning_rate
		self.gamma = discount_rate
		self.epsilon = exploration_rate
		self.beta = decay_rate
		self.sections = sections
		self.size_actions = self.environment.action_space.n
		self.size_observations = len(self.environment.observation_space.low)
		size = [self.sections]*self.size_observations
		size.append(self.size_actions)
		self.q = np.random.uniform(-1, 1, size)
		self.observation_space = []
		self.profit = []
		minimum, maximum = self.aggregator()

		for state in range(self.size_observations):

			start = minimum[state]
			end = maximum[state]
			interval = np.linspace(start, end, self.sections)
			self.observation_space.append(interval)


	def aggregator(self):

		runs = 100
		states = []

		for episode in range(runs):

		    observation = self.environment.reset()
		    done = False

		    while not done:

		        action = self.environment.action_space.sample()
		        observation, reward, done, information = self.environment.step(action)
		        states.append(observation)

		self.environment.close()
		size = len(states)
		maximum = np.zeros(self.size_observations)
		minimum = np.zeros(self.size_observations)

		for entry in range(size):

			for position in range(self.size_observations):

				element = states[entry][position]

				if (entry == 0):
					minimum[position] = element
					maximum[position] = element
				elif (entry > 0):
					if (element > maximum[position]):
						maximum[position] = element
					if (element < minimum[position]):
						minimum[position] = element

		return minimum, maximum


	def sampler(self, sensor):

		encoding = []

		for state in range(self.size_observations):

			variable = sensor[state]
			level = np.digitize(variable, self.observation_space[state])
			encoding.append(level - 1)

		return tuple(encoding)


	def learn(self):

		exploration = 1
		checkpoint = 100
		Reward = []

		for episode in range(self.epochs):

			score, done = 0, False
			observation = self.environment.reset()
			state = self.sampler(observation)

			for step in range(self.maximum_steps):

				exploitation = np.random.uniform(0, 1)

				if (exploitation > exploration):
					action = np.argmax(self.q[state])
				else:
					action = self.environment.action_space.sample()
	
				observation, reward, done, information = self.environment.step(action)
				future_state = self.sampler(observation)
				self.q[state][action] = self.Q(reward, state, action, future_state)
				state = future_state
				score += reward

				if (episode%(checkpoint*10) == 0):
					self.environment.render()
				if done:
					break

			print("Episode:", episode + 1)
			print("Reward:", score, "\n")
			Reward.append(score)
			exploration = self.epsilon + (1 - self.epsilon)*np.exp(-self.beta*episode)

			if (episode%checkpoint == 0):
				average = sum(Reward[-checkpoint:]) / len(Reward[-checkpoint:])
				self.profit.append(average)

		self.environment.close()


	def test(self):

		trials = 100
		failed, passed = 0, 0

		for episode in range(trials):

			observation = self.environment.reset()
			state = self.sampler(observation)
			done, score = False, 0

			while not done:

				self.environment.render()
				action = np.argmax(self.q[state])
				observation, reward, done, information = self.environment.step(action)
				future_state = self.sampler(observation)
				state = future_state
				score += reward

			if (score >= 195):
				passed += 1
			elif (score < 195):
				failed += 1

			print("Episode:", episode)
			print("Score:", score, "\n")

		self.environment.close()

		if (passed == 1):
			print("The agent sucessfully passed {} trial and failed {} attempts.\n".format(passed, failed))
		elif (failed == 1):
			print("The agent sucessfully passed {} trials and failed {} attempt.\n".format(passed, failed))
		else:
			print("The agent sucessfully passed {} trials and failed {} attempts.\n".format(passed, failed))


	def plot(self):

		episodes = range(1, self.epochs + 1, 100)
		plt.figure()
		axis = plt.axes(facecolor = "#E6E6E6")
		axis.set_axisbelow(True)
		plt.grid(color = "w", linestyle = "solid")

		for spine in axis.spines.values():
			spine.set_visible(False)

		plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
		plt.tick_params(axis = "y", which = "both", left = False, right = False)
		plt.plot(episodes, self.profit, color = "blue", linewidth = 1)
		plt.xlabel('Episode')
		plt.ylabel('Average reward per episode')
		plt.savefig('average_reward_over_each_epoch.png', dpi = 200)
		plt.show()
