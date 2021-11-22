// COMMENT: The following code is based on code shown in the library referance for p5.js, which can be found at
// https://p5js.org/reference/#/p5/camera as of writing. The library code contained code for moving around the camera
// viewport using sliders. 

let sliderGroup = [];
let X;
let Y;
let Z;
let centerX;
let centerY;
let centerZ;
let h = 40;
let parsed_neuron_array = [];
let selected_array = 0;
let current_ancestor = "q"
const position_scaling = 2;

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

let selected_neuron_func_bool = "Unselected any";
let selected_func_index = "Unselected any";

for (let i0 = 0; i0 < neuron_function_order.length; i0++) {
  document.getElementById(i0).addEventListener("click", 
  () => {
    selected_neuron_func_bool = true;
    selected_func_index = i0;
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
      console.log(text)
      parsed_neuron_array.push(JSON.parse(text));
    }
    neuron_array.forEach(parse_text);  
  }, false
  );
  reader.readAsText(fileList[0]);
  selected_array = parsed_neuron_array.length-1;
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
  var cnv = createCanvas(1000, 1000, WEBGL);
  cnv.parent("canvasdiv");
}

function draw() {
  background(60);
  noStroke();

}