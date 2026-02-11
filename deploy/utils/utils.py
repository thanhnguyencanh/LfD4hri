from transforms3d import euler
import numpy as np
import xml.etree.ElementTree as ET
import os

class Utils:
    def __init__(self):
        pass

    def find_project_root(self, project_name="DRL"):
        path = os.path.abspath(__file__)
        while True:
            if os.path.basename(path) == project_name:
                return path
            new_path = os.path.dirname(path)
            if new_path == path:
                raise RuntimeError(f"Project root '{project_name}' not found.")
            path = new_path
            return path
    
    def get_config_root_node(self, config_file_name=None, config_file_data=None):
        # get root
        if config_file_data is None:
            with open(config_file_name) as config_file_content:
                config = ET.parse(config_file_content)
            root_node = config.getroot()
        else:
            root_node = ET.fromstring(config_file_data)

        # get root data
        root_data = root_node.get("name")
        assert isinstance(root_data, str)
        root_name = np.array(root_data.split(), dtype=str)

        return root_node, root_name
    
    def read_config_from_node(self, root_node, parent_name, child_name, dtype=int):
        # find parent
        parent_node = root_node.find(parent_name)
        if parent_node is None:
            quit("Parent %s not found" % parent_name)

        # get child data
        child_data = parent_node.get(child_name)
        if child_data is None:
            quit("Child %s not found" % child_name)

        config_val = np.array(child_data.split(), dtype=dtype)
        return config_val