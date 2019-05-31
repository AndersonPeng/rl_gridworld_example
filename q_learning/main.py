from PyQt5.QtWidgets import QApplication
from grid_env import GridEnvironment
from agent import Agent
import sys


if __name__ == "__main__":
	app = QApplication(sys.argv)
	agent = Agent()
	grid_evn = GridEnvironment(name="Q-Learning Example")
	grid_evn.set_agent(agent)
	grid_evn.show()
	
	sys.exit(app.exec_())