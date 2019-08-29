"""
__author__ = "Xingjian Du"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import sys
import os
try:
    sys.path.append(sys.argv[2])
    fn_path = sys.argv[3]
except:
    sys.path.append(sys.argv[1])
    fn_path = sys.argv[2]

    
from config_file import get_config
from agents import *




def main():
    os.system("touch /tmp/debug")
    config = get_config()
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent.name]
    agent = agent_class(config)
    agent.test_one_file((fn_path, None))
    note_fn_list = agent.data_source.test_loader.dataset.note_fn_list
    audio_fn_list = agent.data_source.test_loader.dataset.audio_fn_list
    for pair in zip(audio_fn_list, note_fn_list):
        acc = agent.test_one_file(pair) 
        print(pair, acc)
        '''
        with open("log.csv", "a+") as f:
            print(f"{pair[0]},{pair[1]},{acc}", file=f)
        '''

if __name__ == '__main__':
    main()
