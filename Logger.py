class Logger:
    def __init__(self, output_filepath, ignored_messages_list = []) -> None:
        self.output_filepath = output_filepath
        self.ignored_messages_list = ignored_messages_list

    def log(self, message_type, message):
        if message_type not in self.ignored_messages_list:
            with open(self.output_filepath, 'a') as f:
                f.writelines(message_type + ": " + message)
        
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
    
    