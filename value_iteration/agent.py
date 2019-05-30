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
		self.n_iter   = 1000


	#---------------------------
	# Value iteration algorithm
	#---------------------------
	def value_iteration(self, running):
		for i in range(self.n_iter):
			if not running.is_set(): break			
			self.value_update()

		running.clear()


	#---------------------------
	# Value iteration update once
	#---------------------------
	def value_update(self):
		Q = [0] * 4
		grid_value_tmp = self.grid_env.grid_value.copy()

		#For each state
		for x in range(self.grid_env.n_x):
			for y in range(self.grid_env.n_y):
				#For each action
				for a in range(4):
					Q[a] = 0

					for x_ in range(self.grid_env.n_x):
						for y_ in range(self.grid_env.n_y):
							Q[a] += self.grid_env.grid_trans_prob[x, y, a, x_, y_] * self.grid_env.grid_value[x_, y_]

				#V(s) = max_a{r(s, a) + gamma * sum(P(s'|s, a) * V(s'))}
				grid_value_tmp[x, y] = self.grid_env.grid_reward[x, y] + self.gamma*max(Q)

		self.grid_env.grid_value[:] = grid_value_tmp[:]
		self.grid_env.update()
		self.grid_env.it += 1


	#---------------------------
	# Start a thread for value iteration
	#---------------------------
	def start_value_iteration(self):
		if self.grid_env is None or (self.running is not None and self.running.is_set()):
			return

		self.running = threading.Event()
		self.running.set()
		self.thread = threading.Thread(target=self.value_iteration, args=(self.running,))
		self.thread.start()


	#---------------------------
	# Close the thread for value iteration
	#---------------------------
	def stop_value_iteration(self):
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