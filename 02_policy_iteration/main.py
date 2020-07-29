from PyQt5.QtWidgets import QApplication
from grid_env import GridEnvironment
from agent import Agent
import sys


if __name__ == "__main__":
	app = QApplication(sys.argv)
	agent = Agent()
	grid_env = GridEnvironment(name="Policy Iteration Example")
	grid_env.set_agent(agent)
	grid_env.setFixedSize(1024, 768)
	grid_env.show()
	
	sys.exit(app.exec_())