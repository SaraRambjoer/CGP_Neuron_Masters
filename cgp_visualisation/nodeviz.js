let selected_array = 0;
let current_ancestor = "q"
const node_size = 10;
const x_distance = 125;
const y_distance = 125;
const xoffset = 100;
const yoffset = 750;
const input_col = [255, 0, 0];
const reg_col = [0, 0, 255];
const output_col = [0, 255, 0];

let x_view_offset = 0;
let y_view_offset = 0;

let parsed_neuron_array = [];

let viewModularFunctions = false; // TODO

const neuron_function_order = [
  'axon_birth_program',
  'signal_axon_program',
  'recieve_axon_signal_program',
  'recieve_reward_program',
  'move_program',
  'die_program',
  'neuron_birth_program',
  'action_controller_program',
  'hox_variant_selection_program'
];

const axon_dendrite_function_order = [
  'recieve_signal_neuron_program',
  'recieve_signal_dendrite_program',
  'signal_dendrite_program',
  'signal_neuron_program',
  'accept_connection_program',
  'break_connection_program',
  'recieve_reward_program',
  'die_program',
  'action_controller_program'
];

let selected_neuron_func_bool = true;
let selected_func_index = 0;

let lastDisplay = "None";

for (let i0 = 0; i0 < neuron_function_order.length; i0++) {
  document.getElementById("neuron_" + neuron_function_order[i0]).addEventListener("click", 
  () => {
    console.log("Selected neuron function");
    selected_neuron_func_bool = true;
    selected_func_index = i0;
    document.getElementById("selectedfunc").innerHTML = "Selected func: " + "neuron_" + neuron_function_order[i0];
  }
  );
}

for (let i0 = 0; i0 < axon_dendrite_function_order.length; i0++) {
  document.getElementById("axon_dendrite_" + axon_dendrite_function_order[i0]).addEventListener("click", 
  () => {
    console.log("Selected axon-dendrite function");
    selected_neuron_func_bool = false;
    selected_func_index = i0;
    document.getElementById("selectedfunc").innerHTML = "Selected func: " + "axon_dendrite_" + axon_dendrite_function_order[i0];
  }
  );
}

const inputElement = document.getElementById("neuronfile");
inputElement.addEventListener("change", handleFiles, false);
function handleFiles() {
  const fileList = this.files; /* now you can work with the file list */
  const reader = new FileReader();
  reader.addEventListener("load", () => {
    const neuron_text = reader.result;
    const neuron_array = neuron_text.split("|");
    neuron_array.pop() // remove empty element at the end
    parsed_neuron_array.splice(0, parsed_neuron_array.length); // empty out array
    function parse_text(text) {
      //console.log(text)
      parsed_neuron_array.push(JSON.parse(text));
    }
    neuron_array.forEach(parse_text);  
    selected_array = parsed_neuron_array.length-1;
  }, false
  );
  reader.readAsText(fileList[0]);
}


//document.getElementById("lockid").addEventListener("click", function() {
//  if (parsed_neuron_array.length > 0) {
//    current_ancestor = parsed_neuron_array[selected_array].genome_id;
//    document.getElementById("lockd").innerHTML = current_ancestor;
//  }
//});

document.getElementById("nextdifferent").addEventListener("click", function() {
  if (parsed_neuron_array.length > 0) {
    current_genome_id = parsed_neuron_array[selected_array].genome_id
    for (let i0 = selected_array; i0 < parsed_neuron_array.length; i0++) {
      if (parsed_neuron_array[i0].genome_id != current_genome_id) {
        selected_array = i0;
        break;
      }
    }
  }
});

document.getElementById("prevdifferent").addEventListener("click", function() {
  if (parsed_neuron_array.length > 0) {
    current_genome_id = parsed_neuron_array[selected_array].genome_id
    for (let i0 = selected_array; i0 >= 0; i0--) {
      if (parsed_neuron_array[i0].genome_id != current_genome_id) {
        selected_array = i0;
        break;
      }
    }
  }
});

//document.getElementById("nextsame").addEventListener("click", function() {
//  if (parsed_neuron_array.length > 0) {
//    current_genome_id = parsed_neuron_array[selected_array].genome_id
//    for (let i0 = selected_array+1; i0 < parsed_neuron_array.length; i0++) {
//      if (parsed_neuron_array[i0].genome_id.substring(0, current_ancestor.length) == current_ancestor) {
//        selected_array = i0;
//        break;
//      }
//    }
//  }
//});
//
//document.getElementById("prevsame").addEventListener("click", function() {
//  if (parsed_neuron_array.length > 0) {
//    current_genome_id = parsed_neuron_array[selected_array].genome_id
//    for (let i0 = selected_array-1; i0 >= 0; i0--) {
//      if (parsed_neuron_array[i0].genome_id.substring(0, current_ancestor.length) == current_ancestor) {
//        selected_array = i0;
//        break;
//      }
//    }
//  }
//});


function setup() {
  var cnv = createCanvas(1000, 1000);
  cnv.parent("canvasdiv");
}

function getNodeType(node, nodeTypeList) {
  for (let i0 = 0; i0 < nodeTypeList.length; i0++) {
    if (nodeTypeList[i0][0] == node) {
      return nodeTypeList[i0][1];
    }
  }
}

function getConnectionCoordinates(connection, node_positions) {
  let xFrom = "blank";
  let yFrom = "blank";
  let xTo = "blank";
  let yTo = "blank";
  for (let i0 = 0; i0 < node_positions.length; i0++) {
    if (connection[0] == node_positions[i0][0]) {
      xFrom = node_positions[i0][1];
      yFrom = node_positions[i0][2];
      break;
    }
  }
  for (let i0 = 0; i0 < node_positions.length; i0++) {
    if (connection[1] == node_positions[i0][0]) {
      xTo = node_positions[i0][1];
      yTo = node_positions[i0][2];
      break;
    }
  }
  return [xFrom, yFrom, xTo, yTo];
}

function countArrayElementsEquals(the_array, the_element) {
  let count = 0;
  for (let i0 = 0; i0 < the_array.length; i0++) {
    if (the_array[i0] == the_element) {
      count += 1;
    }
  }
  return count;
}

function draw() {
  background(60);
  noStroke();
  let nodes = [];
  // Select right program in program
  if (parsed_neuron_array.length > 0) {
    let genome = parsed_neuron_array[selected_array];
    const fitness = genome['genome fitness']; // TODO display these
    const id = genome['genome id'];
    document.getElementById("genomeid").innerHTML = "ID: " + id.toString();
    document.getElementById("genomefitness").innerHTML = "Fitness: " + fitness.toString();
    if (viewModularFunctions) {
      // TODO
    }
    else {
      if (selected_neuron_func_bool) {
        if (neuron_function_order[selected_func_index] == "hox_variant_selection_program") {
          genome = genome["hex_selector"];
        }
        else {
          genome = genome[neuron_function_order[selected_func_index]];
        }
      }
      else {
        genome = genome[axon_dendrite_function_order[selected_func_index]];
      }
    }
    // Get order through depth
    let node_depths = [];
    // add input_nodes at depth 0 
    // Add everything in sequencial order
    let visited = [];
    let new_frontier = [];
    let depth = 0;
    
    let input_nodes = [];
    for (let i0 = 0; i0 < genome['input_nodes'].length; i0++) {
      input_nodes.push(JSON.parse(JSON.stringify(genome['input_nodes'][i0][0])));
    }
    let frontier = JSON.parse(JSON.stringify(input_nodes)); // hack
    let output_nodes = [];
    for (let i0 = 0; i0 < genome['output_nodes'].length; i0++) {
      output_nodes.push(JSON.parse(JSON.stringify(genome['output_nodes'][i0][0])));
    }
    while ((frontier.length > 0) || (new_frontier.length > 0)) {
      if (frontier.length == 0) {
        depth = depth + 1;
        frontier = JSON.parse(JSON.stringify(new_frontier));
        new_frontier = [];
      }
      let node = frontier.pop();
      if (!visited.includes(node)) {
        visited.push(node);
        node_depths.push(JSON.parse(JSON.stringify(depth)));
        for (let i0 = 0; i0 < genome['connection_pairs'].length; i0++) {
          const connection = JSON.parse(JSON.stringify(genome['connection_pairs'][i0]));
          if (connection[0] == node) {
            if (genome['active_nodes'].includes(connection[1])) {
              new_frontier.push(connection[1]);
            }
          }
        }
      }
      else {
        const indx = visited.indexOf(node);
        node_depths[indx] = JSON.parse(JSON.stringify(depth));
        node_depths.push(JSON.parse(JSON.stringify(depth)));
        for (let i0 = 0; i0 < genome['connection_pairs'].length; i0++) {
          const connection = JSON.parse(JSON.stringify(genome['connection_pairs'][i0]));
          if (connection[0] == node) {
            if (genome['active_nodes'].includes(connection[1])) {
              new_frontier.push(connection[1]);
            }
          }
        }
      }
    }
    let node_positions = []
    // Paint nodes by depth
    let node_depths_processed = [];
    for (let i0 = 0; i0 < visited.length; i0++) {
      const node = JSON.parse(JSON.stringify(visited[i0]));
      const depth = JSON.parse(JSON.stringify(node_depths[i0]));
      while (node_depths_processed.length <= depth) {
        node_depths_processed.push([]);
      }
      node_depths_processed[depth].push(node);
    }
    for (let i0 = 0; i0 < node_depths_processed.length; i0++) {
      let nodes = JSON.parse(JSON.stringify(node_depths_processed[i0]));
      if (nodes.length != 0) {
        for (let i1 = 0; i1 < nodes.length; i1++) {
          const x = i1 * x_distance + xoffset + x_view_offset;
          const y = -i0 * y_distance + yoffset + y_view_offset;
          const node = nodes[i1];
          let col = "None"
          if (input_nodes.includes(node)) {
            col = input_col;
          }
          else if (output_nodes.includes(node)) {
            col = output_col;
          }
          else {
            col = reg_col;
          }
          const the_text = getNodeType(node, genome['node_types']);
          node_positions.push([node, x, y, the_text, col]);
        }
      }
    }
    let shortened_connection_pairs = [];
    for (let i0 = 0; i0 < genome['connection_pairs'].length; i0++) {
      const connection = JSON.parse(JSON.stringify(genome['connection_pairs'][i0]));
      shortened_connection_pairs.push[[connection[0]], connection[1], connection[2]];
    }

    for (let i0 = 0; i0 < genome['connection_pairs'].length; i0++) {
      const connection = JSON.parse(JSON.stringify(genome['connection_pairs'][i0]));
      const [fromX, fromY, toX, toY] = getConnectionCoordinates(connection, node_positions);
      stroke(color(0, 0, 0));
      strokeWeight(1);
      const connection2 = JSON.parse(JSON.stringify(genome['connection_pairs'][i0]));
      short_local = shortened_connection_pairs.slice(0, i0-1);
      if (i0 == 0) {
        short_local = [];
      }
      fill(255, 255, 255);
      if (short_local.includes([connection2[0], connection2[1], connection2[2]])) {
        const times = countArrayElementsEquals(short_local, [connection2[0], connection2[1], connection2[2]]);
        nodes.push(line(xf+2*times, yf, xt+2*times, yt));
        nodes.push(text(connection[2], xf + (xt-xf)*(9.0-times)/10.0, yf + (yt - yf)*(9.0-times)/10.0));
      }
      else {
        const [xf, yf, xt, yt] = [parseInt(fromX), parseInt(fromY), parseInt(toX), parseInt(toY)];
        nodes.push(line(xf, yf, xt, yt));
        nodes.push(text(connection[2], xf + (xt-xf)*9.0/10.0, yf + (yt - yf)*9.0/10.0));
      }
    }
    noStroke();
    for (let i0 = 0; i0 < node_positions.length; i0++) {
      let [node, x, y, the_text, col] = node_positions[i0];
      fill(col);
      nodes.push(circle(x, y, node_size));
      nodes.push(text(the_text, x+5, y));
      nodes.push(text(node, x+5, y+20));
      if (input_nodes.includes(node)) {
        const indx = genome['input_nodes'][input_nodes.indexOf(node)][1];
        nodes.push(text(indx.toString(), x-5, y-20));
      }
      let local_output_nodes = JSON.parse(JSON.stringify(output_nodes));
      times = 0;
      let last_indx = 0;
      while (local_output_nodes.includes(node)) {
        times += 1
        const indx = genome['output_nodes'][last_indx + local_output_nodes.indexOf(node)][1];
        nodes.push(text(indx.toString(), x-10*times, y-15));
        local_indx = local_output_nodes.indexOf(node);
        local_output_nodes = local_output_nodes.slice(local_indx+1, genome.length);
        last_indx = last_indx+local_indx+1;
      }
    }
  }
}

function keyPressed() {
  const x = JSON.parse(JSON.stringify(keyCode));
  console.log(x);
  if (x == 87) { // w
    console.log("w");
    y_view_offset -= 50;
  }
  if (x == 65) { // a
    console.log("a");
    x_view_offset -= 50;
  }
  if (x == 83) { // s 
    console.log("s");
    y_view_offset += 50;
  }
  if (x == 68) { // d
    console.log("d");
    x_view_offset += 50;
  }
}