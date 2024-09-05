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
    return pred.sub(labels).square().mean()
}

function predict(x_val){
    let xs = tf.tensor1d(x_val);
    return xs.mul(m).add(b);
}

function mousePressed(){
    
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);
    
    x_vals.push(x);
    y_vals.push(y);

}

function draw(){

    tf.tidy(() => {
        if (x_vals.length > 0){
            const ys = tf.tensor1d(y_vals)
            optimizer.minimize(() => loss(predict(x_vals), ys));
        }
    })

    background(0)

    stroke(255)
    strokeWeight(10)
    for (let i = 0; i<x_vals.length; i++){
        let px = map(x_vals[i], 0, 1, 0, width)
        let py = map(y_vals[i], 0, 1, height, 0)
        point(px,py)
    }


    const tfx = [0,1]
    const tfy = tf.tidy(() => predict(tfx))
    
    tfy.print()

    let x1 = map(tfx[0],0,1,0, width)
    let x2 = map(tfx[1],0,1,0, width)

    let lineY = tfy.dataSync()
   
    let y1 = map(lineY[0],0,1,height,0)
    let y2 = map(lineY[1],0,1,height,0)

    tf.dispose(tfy)

    line(x1,y1,x2,y2)
    strokeWeight(2)
}