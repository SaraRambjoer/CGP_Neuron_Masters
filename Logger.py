import os.path
import json
import time
from os import mkdir

class Logger:
    def __init__(self, output_dir, ignored_messages_list = []) -> None:
        self.output_dir = output_dir + str(time.time())
        mkdir(self.output_dir)
        self.ignored_messages_list = ignored_messages_list
        self.message_type_to_filepath = {
            "cgp_function_exec_prio1" : "exec_message_log.txt", 
            "cgp_function_exec_prio2" : "exec_message_log.txt",
            "CGPProgram image" : "cgp_program_image.txt",
            "neuron_image" : "neuron_image.txt",
            "graphlog_instance" : "graphlogs_instances.txt",
            "graphlog_run" : "graphlog_run.txt",
            "setup_info" : "setup_info.txt"
        }

    def log(self, message_type, message):
        if message_type not in self.ignored_messages_list:
            with open(os.path.join(self.output_dir, self.message_type_to_filepath[message_type]), 'a') as f:
                f.writelines(message + "\n")
            self.backlog = []
        
    def log_cgp_program(self, active_nodes, output_nodes):
        node_types = [f"({node.id}, {node.gettype()})" for node in active_nodes]
        connection_pairs = []
        for node in active_nodes:
            for subscriber in node.subscribers:
                connection_pairs.append(f"({node.id}, {subscriber.id})")
        self.log(
            "CGPProgram image",
            "Active nodes: " + ", ".join(node_types) + "\n" +  
            "Connection pairs: " + ", ".join(connection_pairs) + "\n" + 
            "Output_nodes: " + ", ".join([str(node.id) for node in output_nodes])
        )
    
    def log_json(self, message_type, json_data):
        if message_type not in self.ignored_messages_list:
            # Log json
            with open(os.path.join(self.output_dir, self.message_type_to_filepath[message_type]), 'a') as f:
                json.dump(json_data, f)
                f.write("|")