const costData = [{
    monthYear: "January 2022",
    cost: 25.12
},{
    monthYear: "February 2022",
    cost: 22.12
},{
    monthYear: "March 2022",
    cost: 26.12
},{
    monthYear: "April 2022",
    cost: 22.90
},{
    monthYear: "May 2022",
    cost: 19.14
},{
    monthYear: "June 2022",
    cost: 21.82
},{
    monthYear: "July 2022",
    cost: 24.32
},{
    monthYear: "August 2022",
    cost: 22.11
},{
    monthYear: "September 2022",
    cost: 20.21
},{
    monthYear: "October 2022",
    cost: 21.14
},{
    monthYear: "November 2022",
    cost: 19.96
},{
    monthYear: "December 2022",
    cost: 20.52
},{
    monthYear: "January 2023",
    cost: 19.12
},{
    monthYear: "February 2023",
    cost: 21.51
},{
    monthYear: "March 2023",
    cost: 27.13
},{
    monthYear: "April 2023",
    cost: 18.17
},{
    monthYear: "May 2023",
    cost: 16.10
},{
    monthYear: "June 2023",
    cost: 17.45
},{
    monthYear: "July 2023",
    cost: 20.17
},{
    monthYear: "August 2023",
    cost: 19.18
},{
    monthYear: "September 2023",
    cost: 21.17
},{
    monthYear: "October 2023",
    cost: 21.00
},{
    monthYear: "November 2023",
    cost: 28.5
},{
    monthYear: "December 2023",
    cost: 23.12
}]

function monthYearToNumber(monthYearValue){
    const [month, year] = monthYearValue.split(" ")
    const monthIndex = new Date(Date.parse(month + " " +  year)).getMonth()+1;
    return (parseInt(year) - 2022) *12 + monthIndex;   

}

let x_vals = costData.map((date) => monthYearToNumber(date.monthYear));
let y_vals = costData.map((costs) => costs.cost);

let m, b;

const learningRate = 0.001;
const optimizer = tf.train.sgd(learningRate)

function setup(){
    createCanvas(1000,1000);
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

function projectedNextMonth(x_val){
    // const x = tf.variable(tf.scalar(x_val))
    // const val = x.mul(m).add(b)
    // console.log(val.dataSync())
    // return val;
    let m_val = parseFloat(m)
    let b_val = parseFloat(b)

    const x_predict = x_val*m_val+b_val;
    return x_predict
}

function sgd(){

}

// function mousePressed(){

//     let x = map(mouseX, 0, width, 0, 1);
//     let y = map(mouseY, 0, height, 1, 0);

//     x_vals.push(x);
//     y_vals.push(y);

// }

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
        let px = map(x_vals[i], 0, max(x_vals) + 1, min(x_vals), width)
        let py = map(y_vals[i], 0, max(y_vals) + 20, height, min(y_vals))
        point(px,py)
    }


    const tfx = [0,min(x_vals)]
    const tfy = tf.tidy(() => predict(tfx))

    tfy.print()

    let x1 = map(tfx[0], 0, max(x_vals)+ 1,min(x_vals), width)
    let x2 = map(tfx[1], 0, max(x_vals)+ 1,min(x_vals), width)

    let lineY = tfy.dataSync()

    let y1 = map(lineY[0],0, max(y_vals) + 20,height,min(y_vals))
    let y2 = map(lineY[1],0, max(y_vals) + 20,height,min(y_vals))

    tf.dispose(tfy)

    line(x1,y1,x2,y2)
    strokeWeight(2)
}