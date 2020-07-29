import threading
import numpy as np


class Agent():
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self):
		self.grid_env = None
		self.running  = None
		self.thread   = None
		self.gamma    = 0.9
		self.alpha    = 0.1
		self.eps      = 0.8
		self.n_iter   = 100


	#---------------------------
	# Q-learning algorithm
	#---------------------------
	def q_learning(self, running):
		for i in range(self.n_iter):
			if not running.is_set(): break

			last_state = self.grid_env.reset()

			while running.is_set():
				#Epsilon greedy
				if np.random.uniform(0, 1) < self.eps:
					action = np.random.randint(0, 4)
				else:
					action = self.choose_action(last_state)

				q_value             = self.grid_env.grid_q_value[last_state[0], last_state[1], action]
				state, reward, done = self.grid_env.step(action)
				max_next_q_value    = np.max(self.grid_env.grid_q_value[state[0], state[1], :])
				
				#Q(S, A) = Q(S, A) + alpha * [r(S, A) + gamma*max_a(Q(S', a)) - Q(S, A)]
				self.grid_env.grid_q_value[last_state[0], last_state[1], action] = q_value + self.alpha*(reward + self.gamma*max_next_q_value - q_value)
				last_state = state.copy()

				if done: break

		self.grid_env.reset()
		running.clear()


	#---------------------------
	# Q-learning update once
	#---------------------------
	def q_learning_update(self, q_value, max_next_q_value, reward):
		return q_value + self.alpha*(reward + self.gamma*max_next_q_value - q_value)


	#---------------------------
	# Choose action
	#---------------------------
	def choose_action(self, state):
		return np.argmax(self.grid_env.grid_q_value[state[0], state[1], :])


	#---------------------------
	# Perform Q-learning
	# Only 1 iteration
	#---------------------------
	def step_q_learning(self):
		if self.grid_env is None or (self.running is not None and self.running.is_set()):
			return

		n_iter      = self.n_iter
		self.n_iter = 1

		self.running = threading.Event()
		self.running.set()
		self.thread = threading.Thread(target=self.q_learning, args=(self.running,))
		self.thread.start()
		
		self.thread.join()
		self.n_iter = n_iter


	#---------------------------
	# Start a thread for Q-learning
	#---------------------------
	def start_q_learning(self):
		if self.grid_env is None or (self.running is not None and self.running.is_set()):
			return

		self.running = threading.Event()
		self.running.set()
		self.thread = threading.Thread(target=self.q_learning, args=(self.running,))
		self.thread.start()


	#---------------------------
	# Close the thread for Q-learning
	#---------------------------
	def stop_q_learning(self):
		if self.running is not None and self.thread is not None:
			self.running.clear()
			self.thread.join()


	#---------------------------
	# Check is running
	#---------------------------
	def is_running(self):
		if self.running is None:
			return False

		return self.running.is_set()