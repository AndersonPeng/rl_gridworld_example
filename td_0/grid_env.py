from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QLabel, QSlider
from PyQt5.QtGui import QPainter, QBrush, QColor, QPixmap, QFont
from PyQt5.QtCore import Qt, QRect, QTimer
from PyQt5 import QtCore
import numpy as np
import sys
import time


red   = np.array([255, 0, 0])
green = np.array([0, 255, 0])
white = np.array([255, 255, 255])


#QWidget gui class
class GridEnvironment(QWidget):
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self, grid_size=128, n_x=5, n_y=5, name="RL Grid Example"):
		super().__init__()
		
		#Init environment
		self.offset      = 16
		self.grid_size   = grid_size
		self.n_x         = n_x
		self.n_y         = n_y
		self.agent_pos   = [0, 0]
		self.grid_map    = np.zeros((n_x, n_y), dtype=np.int32)
		self.grid_reward = np.zeros((n_x, n_y), dtype=np.float32)
		self.grid_value  = np.zeros((n_x, n_y), dtype=np.float32)
		self.win_reward  = 4
		self.loss_reward = -4
		self.trans_prob  = 0.8
		self.it          = 0
		self.agent       = None

		self.agent_pos  = [2, 3]
		self.last_state = [2, 3]
		self.grid_map[0, 1] = self.grid_map[4, 0] = 1
		self.grid_map[0, 3] = self.grid_map[2, 1] = self.grid_map[4, 2] = -1

		for x in range(n_x):
			for y in range(n_y):
				if self.grid_map[x, y] == 1:
					self.grid_reward[x, y] = self.win_reward
				elif self.grid_map[x, y] == -1:
					self.grid_reward[x, y] = self.loss_reward
		
		#Init Qt gui
		self.agent_pixmap = QPixmap("../resource/agent.png")
		self.goal_pixmap  = QPixmap("../resource/goal.png")
		self.hole_pixmap  = QPixmap("../resource/hole.png")
		self.font         = QFont("Consolus", 12, QFont.Bold)
		self.slider_style = """QSlider::groove:horizontal 
			{
				border: 1px solid #bbb;
				background: white;
				height: 10px;
				border-radius: 4px;
			}
			QSlider::sub-page:horizontal 
			{
    			background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #66e, stop: 1 #bbf);
    			background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1, stop: 0 #bbf, stop: 1 #55f);
    			border: 1px solid #777;
    			height: 10px;
    			border-radius: 4px;
			}
			QSlider::add-page:horizontal 
			{
				background: #fff;
				border: 1px solid #777;
				height: 10px;
				border-radius: 4px;
			}
			QSlider::handle:horizontal {
				background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #eee, stop: 1 #ccc);
				border: 1px solid #777;
				width: 13px;
				margin-top: -2px;
				margin-bottom: -2px;
				border-radius: 4px;
			}"""

		self.ui_timer = QTimer()
		self.ui_timer.timeout.connect(self.update_ui)
		self.ui_timer.start(200)

		self.setGeometry(200, 200, 1024, 768)
		self.setWindowTitle(name)

		self.start_btn = QPushButton("Start", parent=self)
		self.start_btn.setGeometry(2*self.offset + self.n_x*self.grid_size, self.offset, 128, 64)
		self.start_btn.setFont(self.font)

		self.stop_btn = QPushButton("Stop", parent=self)
		self.stop_btn.setGeometry(3*self.offset + self.n_x*self.grid_size + 128, self.offset, 128, 64)
		self.stop_btn.setFont(self.font)

		self.step_btn = QPushButton("Step", parent=self)
		self.step_btn.setGeometry(2*self.offset + self.n_x*self.grid_size, 2*self.offset + 64, 128, 64)
		self.step_btn.setFont(self.font)

		self.clear_btn = QPushButton("Clear", parent=self)
		self.clear_btn.setGeometry(3*self.offset + self.n_x*self.grid_size + 128, 2*self.offset + 64, 128, 64)
		self.clear_btn.setFont(self.font)
		self.clear_btn.clicked.connect(self.clear_grid_value)

		self.win_reward_label = QLabel("Win reward: ", parent=self)
		self.win_reward_label.setGeometry(2*self.offset + self.n_x*self.grid_size, 3*self.offset + 128, 128, 64)
		self.win_reward_label.setFont(self.font)
		self.win_reward_line = QLineEdit("{:.2f}".format(self.win_reward), parent=self)
		self.win_reward_line.setGeometry(3*self.offset + self.n_x*self.grid_size + 128, 3*self.offset + 128, 128, 64)
		self.win_reward_line.setFont(self.font)

		self.loss_reward_label = QLabel("Lose reward: ", parent=self)
		self.loss_reward_label.setGeometry(2*self.offset + self.n_x*self.grid_size, 4*self.offset + 192, 128, 64)
		self.loss_reward_label.setFont(self.font)
		self.loss_reward_line = QLineEdit("{:.2f}".format(self.loss_reward), parent=self)
		self.loss_reward_line.setGeometry(3*self.offset + self.n_x*self.grid_size + 128, 4*self.offset + 192, 128, 64)
		self.loss_reward_line.setFont(self.font)

		self.set_btn = QPushButton("Set", parent=self)
		self.set_btn.setGeometry(2*self.offset + self.n_x*self.grid_size, 5*self.offset + 256, 128, 64)
		self.set_btn.setFont(self.font)
		self.set_btn.clicked.connect(self.update_reward)

		self.gamma_label = QLabel("Gamma: 0.00", parent=self)
		self.gamma_label.setGeometry(2*self.offset + self.n_x*self.grid_size, 6*self.offset + 320, 180, 64)
		self.gamma_label.setFont(self.font)
		self.gamma_slider = QSlider(Qt.Horizontal, parent=self)
		self.gamma_slider.setGeometry(3*self.offset + self.n_x*self.grid_size + 180, 6*self.offset + 320, 128, 64)
		self.gamma_slider.setFont(self.font)
		self.gamma_slider.setMinimum(0)
		self.gamma_slider.setMaximum(100)
		self.gamma_slider.setTickInterval(1)
		self.gamma_slider.setValue(0)
		self.gamma_slider.setStyleSheet(self.slider_style)
		self.gamma_slider.valueChanged.connect(self.update_gamma)

		self.alpha_label = QLabel("Alpha: 0.00", parent=self)
		self.alpha_label.setGeometry(2*self.offset + self.n_x*self.grid_size, 7*self.offset + 384, 180, 64)
		self.alpha_label.setFont(self.font)
		self.alpha_slider = QSlider(Qt.Horizontal, parent=self)
		self.alpha_slider.setGeometry(3*self.offset + self.n_x*self.grid_size + 180, 7*self.offset + 384, 128, 64)
		self.alpha_slider.setMinimum(0)
		self.alpha_slider.setMaximum(100)
		self.alpha_slider.setTickInterval(1)
		self.alpha_slider.setValue(0)
		self.alpha_slider.setStyleSheet(self.slider_style)
		self.alpha_slider.valueChanged.connect(self.update_alpha)

		self.eps_label = QLabel("Epsilon: 0.00", parent=self)
		self.eps_label.setGeometry(2*self.offset + self.n_x*self.grid_size, 8*self.offset + 448, 180, 64)
		self.eps_label.setFont(self.font)
		self.eps_slider = QSlider(Qt.Horizontal, parent=self)
		self.eps_slider.setGeometry(3*self.offset + self.n_x*self.grid_size + 180, 8*self.offset + 448, 128, 64)
		self.eps_slider.setMinimum(0)
		self.eps_slider.setMaximum(100)
		self.eps_slider.setTickInterval(1)
		self.eps_slider.setValue(0)
		self.eps_slider.setStyleSheet(self.slider_style)
		self.eps_slider.valueChanged.connect(self.update_eps)

		self.trans_label = QLabel("Trans. Prob.: {:.2f}".format(self.trans_prob), parent=self)
		self.trans_label.setGeometry(2*self.offset + self.n_x*self.grid_size, 9*self.offset + 512, 180, 64)
		self.trans_label.setFont(self.font)
		self.trans_slider = QSlider(Qt.Horizontal, parent=self)
		self.trans_slider.setGeometry(3*self.offset + self.n_x*self.grid_size + 180, 9*self.offset + 512, 128, 64)
		self.trans_slider.setMinimum(0)
		self.trans_slider.setMaximum(100)
		self.trans_slider.setTickInterval(1)
		self.trans_slider.setValue(self.trans_prob * 100)
		self.trans_slider.setStyleSheet(self.slider_style)
		self.trans_slider.valueChanged.connect(self.update_trans_prob)

		self.it_label = QLabel("Iteration: {:d}".format(self.it), parent=self)
		self.it_label.setGeometry(2*self.offset + self.n_x*self.grid_size, 9*self.offset + 576, 256, 64)
		self.it_label.setFont(self.font)


	#---------------------------
	# Reset
	#---------------------------
	def reset(self):
		self.agent_pos[0] = 2
		self.agent_pos[1] = 3
		return self.agent_pos


	#---------------------------
	# Step
	#---------------------------
	def step(self, action):
		#Environment transition probability
		rand_val = np.random.uniform(0, 1)
		tmp      = (1 - self.trans_prob) / 2.

		if rand_val < tmp:
			if action == 0:   action = 2	#up to left
			elif action == 1: action = 3	#down to right
			elif action == 2: action = 1	#left to down
			elif action == 3: action = 0	#right to up
		elif rand_val > 1 - tmp:
			if action == 0:   action = 3	#up to right
			elif action == 1: action = 2	#down to left
			elif action == 2: action = 0	#left to up
			elif action == 3: action = 1	#right to down

		#Action 0: Up
		if action == 0:
			self.agent_pos[1] -= 1
			if self.agent_pos[1] < 0:
				self.agent_pos[1] = 0

		#Action 1: Down
		elif action == 1:
			self.agent_pos[1] += 1
			if self.agent_pos[1] >= self.n_y:
				self.agent_pos[1] = self.n_y - 1
		
		#Action 2: Left
		elif action == 2:
			self.agent_pos[0] -= 1
			if self.agent_pos[0] < 0:
				self.agent_pos[0] = 0

		#Action 3: Right
		elif action == 3:
			self.agent_pos[0] += 1
			if self.agent_pos[0] >= self.n_x:
				self.agent_pos[0] = self.n_x - 1

		#Get next step
		reward = self.grid_reward[self.agent_pos[0], self.agent_pos[1]]

		if self.grid_map[self.agent_pos[0], self.agent_pos[1]] != 0:
			done = True
			self.it += 1
		else:
			done = False

		self.update()

		return self.agent_pos, reward, done


	#---------------------------
	# Draw a rectangle
	#---------------------------
	def draw_rect(self, painter, color=QColor(255, 255, 255), rect=QRect(10, 10, 50, 50)):
		painter.setBrush(color)
		painter.drawRect(rect)


	#---------------------------
	# Set agent
	#---------------------------
	def set_agent(self, agent):
		if self.agent is not None:
			self.agent.grid_env = None
			self.start_btn.clicked.disconnect(self.agent.start_tabular_td)
			self.stop_btn.clicked.disconnect(self.agent.stop_tabular_td)
			self.step_btn.clicked.disconnect(self.agent.step_tabular_td)

		self.agent = agent
		self.agent.grid_env = self
		self.start_btn.clicked.connect(self.agent.start_tabular_td)
		self.stop_btn.clicked.connect(self.agent.stop_tabular_td)
		self.step_btn.clicked.connect(self.agent.step_tabular_td)
		self.gamma_slider.setValue(self.agent.gamma * 100)
		self.alpha_slider.setValue(self.agent.alpha * 100)
		self.eps_slider.setValue(self.agent.eps * 100)


	#---------------------------
	# Clear grid value
	#---------------------------
	def clear_grid_value(self):
		for x in range(self.n_x):
			for y in range(self.n_y):
				self.grid_value[x, y] = 0

		self.it = 0
		self.update()


	#---------------------------
	# Update reward
	#---------------------------
	def update_reward(self):
		try:
			self.win_reward  = float(self.win_reward_line.text())
			self.loss_reward = float(self.loss_reward_line.text())
		except:
			print("ERROR: Failed to convert string to float")

		for x in range(self.n_x):
			for y in range(self.n_y):
				if self.grid_map[x, y] == 1:
					self.grid_reward[x, y] = self.win_reward
				elif self.grid_map[x, y] == -1:
					self.grid_reward[x, y] = self.loss_reward

		self.update()


	#---------------------------
	# Update alpha
	#---------------------------
	def update_alpha(self):
		if self.agent is None: return

		self.agent.alpha = self.alpha_slider.value() / 100.0
		self.alpha_label.setText("Alpha: {:.2f}".format(self.agent.alpha))


	#---------------------------
	# Update gamma
	#---------------------------
	def update_gamma(self):
		if self.agent is None: return

		self.agent.gamma = self.gamma_slider.value() / 100.0
		self.gamma_label.setText("Gamma: {:.2f}".format(self.agent.gamma))


	#---------------------------
	# Update epsilon
	#---------------------------
	def update_eps(self):
		if self.agent is None: return

		self.agent.eps = self.eps_slider.value() / 100.0
		self.eps_label.setText("Epsilon: {:.2f}".format(self.agent.eps))


	#---------------------------
	# Update transition probability
	#---------------------------
	def update_trans_prob(self):
		self.trans_prob = self.trans_slider.value() / 100.0
		self.trans_label.setText("Trans. Prob.: {:.2f}".format(self.trans_prob))


	#---------------------------
	# Update UI
	#---------------------------
	def update_ui(self):
		self.it_label.setText("Iteration: {:d}".format(self.it))


	#---------------------------
	# Qt paint event
	#---------------------------
	def paintEvent(self, e):
		painter = QPainter()
		painter.begin(self)
		painter.setFont(self.font)
 
 		#For each grid
		for x in range(self.n_x):
			for y in range(self.n_y):
				#Draw goal
				if self.grid_map[x, y] == 1:
					if self.win_reward > 0:
						c = QColor(0, 255, 0)
					elif self.win_reward < 0:
						c = QColor(255, 0, 0)
					else:
						c = QColor(255, 255, 255)

					self.draw_rect(
						painter,
						color=c, 
						rect=QRect(self.offset + x*self.grid_size, self.offset + y*self.grid_size, self.grid_size, self.grid_size)
					)
					painter.drawText(
						QRect(self.offset + x*self.grid_size, self.offset + y*self.grid_size, self.grid_size, self.grid_size), 
						Qt.AlignCenter,
						"{:.2f}".format(self.grid_reward[x, y])
					)
					painter.drawPixmap(QRect(
						self.offset + x*self.grid_size, 
						self.offset + y*self.grid_size, 
						self.grid_size, 
						self.grid_size
					), self.goal_pixmap)

				#Draw hole
				elif self.grid_map[x, y] == -1:
					if self.loss_reward > 0:
						c = QColor(0, 255, 0)
					elif self.loss_reward < 0:
						c = QColor(255, 0, 0)
					else:
						c = QColor(255, 255, 255)

					self.draw_rect(
						painter,
						color=c, 
						rect=QRect(self.offset + x*self.grid_size, self.offset + y*self.grid_size, self.grid_size, self.grid_size)
					)
					painter.drawText(
						QRect(self.offset + x*self.grid_size, self.offset + y*self.grid_size, self.grid_size, self.grid_size), 
						Qt.AlignCenter,
						"{:.2f}".format(self.grid_reward[x, y])
					)
					painter.drawPixmap(QRect(
						self.offset + x*self.grid_size, 
						self.offset + y*self.grid_size, 
						self.grid_size, 
						self.grid_size
					), self.hole_pixmap)

				#Draw grid
				else:
					if self.grid_value[x, y] < 0:
						if self.loss_reward == 0:
							alpha = 0
						else:
							alpha = self.grid_value[x, y] / self.loss_reward
							if alpha > 1: alpha = 1.

						c = red * alpha + white * (1. - alpha)
					else:
						if self.win_reward == 0:
							alpha = 0
						else:
							alpha = self.grid_value[x, y] / self.win_reward
							if alpha > 1: alpha = 1.

						c = green * alpha + white * (1. - alpha)

					self.draw_rect(
						painter,
						color=QColor(int(c[0]), int(c[1]), int(c[2])), 
						rect=QRect(self.offset + x*self.grid_size, self.offset + y*self.grid_size, self.grid_size, self.grid_size)
					)
					painter.drawText(
						QRect(self.offset + x*self.grid_size, self.offset + y*self.grid_size, self.grid_size, self.grid_size), 
						Qt.AlignCenter,
						"{:.2f}".format(self.grid_value[x, y])
					)

		#Draw agent
		painter.drawPixmap(QRect(
			self.offset + self.agent_pos[0]*self.grid_size, 
			self.offset + self.agent_pos[1]*self.grid_size, 
			self.grid_size, 
			self.grid_size
		), self.agent_pixmap)

		painter.end()


	#---------------------------
	# Qt key press event
	#---------------------------
	def keyPressEvent(self, e):
		if self.agent is not None and self.agent.is_running():
			return

		key = e.key()

		if key == QtCore.Qt.Key_W:
			state, reward, done = self.step(0)

		elif key == QtCore.Qt.Key_S:
			state, reward, done = self.step(1)
		
		elif key == QtCore.Qt.Key_A:
			state, reward, done = self.step(2)

		elif key == QtCore.Qt.Key_D:
			state, reward, done = self.step(3)
		
		else:
			return

		value      = self.grid_value[state[0], state[1]]
		last_value = self.grid_value[self.last_state[0], self.last_state[1]]

		self.grid_value[self.last_state[0], self.last_state[1]] = self.agent.tabular_td_update(last_value, value, reward)
		self.last_state = state.copy()

		if done:
			self.last_state = self.reset()

		self.update()


	#---------------------------
	# Qt close event
	#---------------------------
	def closeEvent(self, e):
		if self.agent is not None:
			self.agent.stop_tabular_td()