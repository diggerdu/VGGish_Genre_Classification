"""
__author__ = "Xingjian Du"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import sys
try:
    sys.path.append(sys.argv[2])
except:
    sys.path.append(sys.argv[1])

    
from config_file import get_config
from agents import *




def main():
    config = get_config()
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent.name]
    agent = agent_class(config)
    loss = agent.evaluate(sep_only=True)
    __import__('pdb').set_trace() 
    #agent.run(mode='train')
    #agent.finalize()


if __name__ == '__main__':
    main()
