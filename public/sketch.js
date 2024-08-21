

function setup(){
    const tf = require('@tensorflow/tfjs');

    noCanvas();
    const model = tf.tensor2d([2,23,4,15,6,6], [2,3], 'int32')

    console.log(model.data().then(
        function(data){
            console.log(data);
        }
    ));

}

// function draw(){

// }