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
		self.n_iter   = 100


	#---------------------------
	# Policy iteration algorithm
	#---------------------------
	def policy_iteration(self, running):
		for i in range(self.n_iter):
			if not running.is_set(): break			
			self.policy_update()

		running.clear()


	#---------------------------
	# Policy update
	#---------------------------
	def policy_update(self):
		self.policy_evaluation()
		self.policy_improvement()
		self.grid_env.it += 1


	#---------------------------
	# Policy evaluation
	#---------------------------
	def policy_evaluation(self):
		grid_value_tmp = self.grid_env.grid_value.copy()

		#For each state
		for x in range(self.grid_env.n_x):
			for y in range(self.grid_env.n_y):
				#a = pi(s)
				a = self.grid_env.grid_policy[x, y]

				#V(s) = sum_s'{P(s'|s, a)[r(s, a) + gamma * V(s')]}
				grid_value_tmp[x, y] = 0

				for x_ in range(self.grid_env.n_x):
					for y_ in range(self.grid_env.n_y):
						grid_value_tmp[x, y] += self.grid_env.grid_trans_prob[x, y, a, x_, y_] * (self.grid_env.grid_reward[x_, y_] + self.gamma*self.grid_env.grid_value[x_, y_])

		self.grid_env.grid_value[:] = grid_value_tmp[:]
		self.grid_env.update()
		

	#---------------------------
	# Policy improvement
	#---------------------------
	def policy_improvement(self):
		#For each state
		for x in range(self.grid_env.n_x):
			for y in range(self.grid_env.n_y):
				self.grid_env.grid_policy[x, y] = self.choose_action((x, y))

		self.grid_env.update()


	#---------------------------
	# Choose action according to value
	#---------------------------
	def choose_action(self, state):
		# a = argmax_a{sum_s'{P(s'|s, a)[r(s, a) + gamma * V(s')]}}
		Q = np.zeros(4)
		x = state[0]
		y = state[1]

		#Up
		if y - 1 >= 0:
			if self.grid_env.grid_map[x, y-1] == 1:
				return 0
			elif self.grid_env.grid_map[x, y-1] == 0:
				for x_ in range(self.grid_env.n_x):
					for y_ in range(self.grid_env.n_y):
						Q[0] += self.grid_env.grid_trans_prob[x, y, 0, x_, y_] * (self.grid_env.grid_reward[x_, y_] + self.gamma*self.grid_env.grid_value[x_, y_])

		#Down
		if y + 1 < self.grid_env.n_y:
			if self.grid_env.grid_map[x, y+1] == 1:
				return 1
			elif self.grid_env.grid_map[x, y+1] == 0:
				for x_ in range(self.grid_env.n_x):
					for y_ in range(self.grid_env.n_y):
						Q[1] += self.grid_env.grid_trans_prob[x, y, 1, x_, y_] * (self.grid_env.grid_reward[x_, y_] + self.gamma*self.grid_env.grid_value[x_, y_])

		#Left
		if x - 1 >= 0:
			if self.grid_env.grid_map[x-1, y] == 1:
				return 2
			elif self.grid_env.grid_map[x-1, y] == 0:
				for x_ in range(self.grid_env.n_x):
					for y_ in range(self.grid_env.n_y):
						Q[2] += self.grid_env.grid_trans_prob[x, y, 2, x_, y_] * (self.grid_env.grid_reward[x_, y_] + self.gamma*self.grid_env.grid_value[x_, y_])

		#Right
		if x + 1 < self.grid_env.n_x:
			if self.grid_env.grid_map[x+1, y] == 1:
				return 3
			elif self.grid_env.grid_map[x+1, y] == 0:
				for x_ in range(self.grid_env.n_x):
					for y_ in range(self.grid_env.n_y):
						Q[3] += self.grid_env.grid_trans_prob[x, y, 3, x_, y_] * (self.grid_env.grid_reward[x_, y_] + self.gamma*self.grid_env.grid_value[x_, y_])

		return np.argmax(Q)


	#---------------------------
	# Start a thread for policy iteration
	#---------------------------
	def start_policy_iteration(self):
		if self.grid_env is None or (self.running is not None and self.running.is_set()):
			return

		self.running = threading.Event()
		self.running.set()
		self.thread = threading.Thread(target=self.policy_iteration, args=(self.running,))
		self.thread.start()


	#---------------------------
	# Close the thread for policy iteration
	#---------------------------
	def stop_policy_iteration(self):
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