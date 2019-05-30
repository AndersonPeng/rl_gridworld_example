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
				#If non-terminal
				if self.grid_env.grid_map[x, y] == 0:
					for a in range(4):
						Q[a] = 0

						for (x_, y_) in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
							if self.grid_env.valid(x_, y_):
								Q[a] += self.grid_env.get_trans_prob(x, y, a, x_, y_) * self.grid_env.grid_value[x_, y_]

					grid_value_tmp[x, y] = self.grid_env.grid_reward[x, y] + self.gamma*max(Q)

				#If terminal
				else:
					grid_value_tmp[x, y] = self.grid_env.grid_reward[x, y]

		self.grid_env.grid_value[:] = grid_value_tmp[:]
		self.grid_env.update()
		self.grid_env.it += 1


	#---------------------------
	# Perform value iteration
	# Only 1 iteration
	#---------------------------
	def step_value_iteration(self):
		if self.grid_env is None or (self.running is not None and self.running.is_set()):
			return

		n_iter      = self.n_iter
		self.n_iter = 1

		self.running = threading.Event()
		self.running.set()
		self.thread = threading.Thread(target=self.value_iteration, args=(self.running,))
		self.thread.start()

		self.thread.join()
		self.n_iter = n_iter


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