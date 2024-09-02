let x_vals = [];
let y_vals = [];

let m, b;

const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate)

function setup(){
    createCanvas(400,400);
    background(0);

    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}

function loss(pred, labels){
    return pred.sub(labels).squared().mean()
}

function predict(x_vals){
    const xs = tensor1d(x_vals);
    const y_vals = xs.mul(m).add(b)
    console.log(y_vals.data())
    return y_vals = xs.mul(m).add(b);
}

function mousePressed(){
    
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);
    
    x_vals.push(x);
    y_vals.push(y);


}

function draw(){

    const ys = tf.tensor1d(y_vals)
    optimizer.minimize(() => {loss(predict(x_vals), ys)});

    background(0)

    stroke(255)
    strokeWeight(4)
    for (let i = 0; i<x_vals.length; i++){
        let px = map(x_vals[i], 0, 1, 0, width)
        let py = map(y_vals[i], 0, 1, height, 1)
        point(px,py)
    }
}