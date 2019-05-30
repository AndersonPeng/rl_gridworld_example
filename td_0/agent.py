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
		self.gamma    = 0.90
		self.alpha    = 0.1
		self.eps      = 0.8
		self.n_iter   = 10000


	#---------------------------
	# Tabular TD(0) algorithm
	#---------------------------
	def tabular_td(self, running):
		for i in range(self.n_iter):
			if not running.is_set(): break

			last_state = self.grid_env.reset()

			while running.is_set():
				#Epsilon greedy
				if np.random.uniform(0, 1) < self.eps:
					state, reward, done = self.grid_env.step(np.random.randint(0, 4))
				else:
					state, reward, done = self.grid_env.step(self.choose_action(last_state))

				value      = self.grid_env.grid_value[state[0], state[1]]
				last_value = self.grid_env.grid_value[last_state[0], last_state[1]]

				self.grid_env.grid_value[last_state[0], last_state[1]] = last_value + self.alpha*(reward + self.gamma*value - last_value)
				last_state = state.copy()

				if done: break

		self.grid_env.reset()
		running.clear()


	#---------------------------
	# Tabular TD(0) update once
	#---------------------------
	def tabular_td_update(self, last_value, value, reward):
		return last_value + self.alpha*(reward + self.gamma*value - last_value)


	#---------------------------
	# Choose action
	#---------------------------
	def choose_action(self, state):
		values = np.array([-1024, -1024, -1024, -1024], dtype=np.float32)

		#Up
		if state[1] - 1 >= 0:
			if self.grid_env.grid_map[state[0], state[1]-1] == 1:
				return 0
			elif self.grid_env.grid_map[state[0], state[1]-1] == 0:
				values[0] = self.grid_env.grid_value[state[0], state[1]-1]

		#Down
		if state[1] + 1 < self.grid_env.n_y:
			if self.grid_env.grid_map[state[0], state[1]+1] == 1:
				return 1
			elif self.grid_env.grid_map[state[0], state[1]+1] == 0:
				values[1] = self.grid_env.grid_value[state[0], state[1]+1]

		#Left
		if state[0] - 1 >= 0:
			if self.grid_env.grid_map[state[0]-1, state[1]] == 1:
				return 2
			elif self.grid_env.grid_map[state[0]-1, state[1]] == 0:
				values[2] = self.grid_env.grid_value[state[0]-1, state[1]]

		#Right
		if state[0] + 1 < self.grid_env.n_x:
			if self.grid_env.grid_map[state[0]+1, state[1]] == 1:
				return 3
			elif self.grid_env.grid_map[state[0]+1, state[1]] == 0:
				values[3] = self.grid_env.grid_value[state[0]+1, state[1]]

		return np.argmax(values)


	#---------------------------
	# Perform Tabular TD(0)
	# Only 1 iteration
	#---------------------------
	def step_tabular_td(self):
		if self.grid_env is None or (self.running is not None and self.running.is_set()):
			return

		n_iter      = self.n_iter
		self.n_iter = 1

		self.running = threading.Event()
		self.running.set()
		self.thread = threading.Thread(target=self.tabular_td, args=(self.running,))
		self.thread.start()
		
		self.thread.join()
		self.n_iter = n_iter


	#---------------------------
	# Start a thread for Tabular TD(0)
	#---------------------------
	def start_tabular_td(self):
		if self.grid_env is None or (self.running is not None and self.running.is_set()):
			return

		self.running = threading.Event()
		self.running.set()
		self.thread = threading.Thread(target=self.tabular_td, args=(self.running,))
		self.thread.start()


	#---------------------------
	# Close the thread for Tabular TD(0)
	#---------------------------
	def stop_tabular_td(self):
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