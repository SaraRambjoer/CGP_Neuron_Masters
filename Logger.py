import os.path
import json
import time
import yaml
from os import mkdir

class Logger:
    def __init__(self, output_dir, ignored_messages_list = [], enabled = True) -> None:
        self.output_dir = output_dir + str(time.time())
        mkdir(self.output_dir)
        self.ignored_messages_list = ignored_messages_list
        self.intermediary_output_dir = ""
        self.intermediary_intermediary_output_dir = ""
        self.message_type_to_filepath = {
            "cgp_function_exec_prio1" : "exec_message_log.txt", 
            "cgp_function_exec_prio2" : "exec_message_log.txt",
            "CGPProgram image" : "cgp_program_image.txt",
            "neuron_image" : "neuron_image.txt",
            "graphlog_instance" : "graphlogs_instances.txt",
            "graphlog_run" : "graphlog_run.txt",
            "setup_info" : "setup_info.txt"
        }
        self.buffer = []
        self.enabled = enabled

    def log_statistic_data(self, statistic_data):
        with open(os.path.join(self.output_dir, "statistics.yml"), 'w') as f:
            yaml.dump(statistic_data, f)

    def log(self, message_type, message):
        if self.enabled: 
            if message_type not in self.ignored_messages_list:
                if message_type == "instance_end":
                    if len(self.buffer) > 0:
                        with open(self.intermediary_intermediary_output_dir + "/rundat.txt", 'a') as f:
                            f.writelines("\n".join(self.buffer))
                            self.buffer = []
                elif message_type == "run_start":
                    self.intermediary_output_dir = self.output_dir + f"/{message[0]}"
                    mkdir(self.intermediary_output_dir)
                    self.intermediary_output_dir = self.output_dir + f"/{message[0]}/{message[1]}"
                    mkdir(self.intermediary_output_dir)
                elif message_type == "instance_start":
                    self.intermediary_intermediary_output_dir = self.intermediary_output_dir + f"/{message}"
                    mkdir(self.intermediary_intermediary_output_dir)
                elif message_type == "engine_action" or message_type == "instance_solution" or message_type == "instance_results" or message_type == "reward_phase" or message_type == "run_end":
                    self.buffer.append(f"{message}")
                else:
                    with open(os.path.join(self.output_dir, self.message_type_to_filepath[message_type]), 'a') as f:
                        f.writelines(message + "\n")
            
    def log_cgp_program(self, active_nodes, output_nodes):
        if self.enabled:
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
        if self.enabled:
            if message_type not in self.ignored_messages_list:
                # Log json
                with open(os.path.join(self.output_dir, self.message_type_to_filepath[message_type]), 'a') as f:
                    json.dump(json_data, f)
                    f.write("|")