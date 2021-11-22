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

  // from https://github.com/freshfork/p5.RoverCam
  rover = createRoverCam();
  //rover.usePointerLock();    // optional; default is keyboard control only
  rover.setState({           // optional
    position: [-400,-200,-200],
    rotation: [0.4,0.3,0],
    sensitivity: 0.1,
    speed: 0.5
  });

  //create sliders
  //for (var i = 0; i < 6; i++) {
  //  if (i === 2) {
  //    sliderGroup[i] = createSlider(10, 400, 200);
  //  } else {
  //    sliderGroup[i] = createSlider(-400, 400, 0);
  //  }
  //  h = map(i, 0, 6, 5, 200);
  //  sliderGroup[i].position(1100, h);
  //  sliderGroup[i].style('width', '80px');
  //}
}

function draw() {
  background(60);
  // assigning sliders' value to each parameters
  //X = sliderGroup[0].value();
  //Y = sliderGroup[1].value();
  //Z = sliderGroup[2].value();
  //centerX = sliderGroup[3].value();
  //centerY = sliderGroup[4].value();
  //centerZ = sliderGroup[5].value();
  //camera(X, Y, Z, centerX, centerY, centerZ, 0, 1, 0);
  noStroke();
  fill(255, 102, 94);
  let spheres = []
  if (parsed_neuron_array.length > 0) {
    const neudict = parsed_neuron_array[selected_array]; //#* fix selection of selected array
    for (let i0 = 0; i0 < neudict.neurons.length; i0++) {
      const neuron_id = neudict.neurons[i0];
      if (neudict.inputs.includes(neuron_id)) {
        fill(255, 0, 0);
      }
      else if (neudict.outputs.includes(neuron_id)) {
        fill(0, 255, 0);
      }
      else {
        fill(0, 0, 255)
      }
      let position = "nada"
      for (let i1 = 0; i1 < neudict.neuron_id_to_pos.length; i1++) {
        const current_tuple = neudict.neuron_id_to_pos[i1];
        if (current_tuple[0] === neuron_id) {
          position = current_tuple[1];
        }
      }
      push();
      translate(position[0]*position_scaling, position[1]*position_scaling, position[2]*position_scaling);
      spheres.push(sphere(1));
      pop();
      //spheres[spheres.length-1].position = position;
    }

    document.getElementById("run").innerHTML = "Run number: " + neudict["run number"];
    document.getElementById("iteration").innerHTML = "Iteration: " + neudict.iteration.toString()
    document.getElementById("genome").innerHTML = "Genome id: " + neudict.genome_id.toString()

    for (let i0 = 0; i0 < neudict.connections.length; i0++) {
      from_id = neudict.connections[i0][0];
      to_id = neudict.connections[i0][1];
      let frompos = 0
      let topos = 0
      for (let i1 = 0; i1 < neudict.neuron_id_to_pos.length; i1++) {
        const current_tuple = neudict.neuron_id_to_pos[i1];
        if (current_tuple[0] === from_id) {
          frompos = current_tuple[1];
        }
        if (current_tuple[0] == to_id) {
          topos = current_tuple[1];
        }
      }
      strokeWeight(2);
      stroke(color(0, 0, 0));
      spheres.push(line(frompos[0], frompos[1], frompos[2], topos[0], topos[1], topos[2]));
      //midway = (frompos[0] + (topos[0] - frompos[0])*0.5, frompos[1] + (topos[1] - frompos[1]) * 0.5, frompos[2] + (topos[2] - frompos[2]) * 0.5);
      //strokeWeight(5);
      //stroke(color(0, 0, 0));
      //line(frompos[0]*position_scaling, frompos[1]*position_scaling, frompos[2]*position_scaling, midway[0]*position_scaling, midway[1]*position_scaling, midway[2]*position_scaling);
      //stroke(color(0, 255, 255));
      //line(midway[0]*position_scaling, midway[1]*position_scaling, midway[2]*position_scaling, topos[0]*position_scaling, topos[1]*position_scaling, topos[2]*position_scaling);
    }
  }


  //console.log(parsed_neuron_array);

}