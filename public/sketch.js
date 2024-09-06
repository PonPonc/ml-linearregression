const costData = [
    { monthYear: "January 2022", cost: 25.12 },
    { monthYear: "February 2022", cost: 22.12 },
    { monthYear: "March 2022", cost: 26.12 },
    { monthYear: "April 2022", cost: 22.90 },
    { monthYear: "May 2022", cost: 19.14 },
    { monthYear: "June 2022", cost: 21.82 },
    { monthYear: "July 2022", cost: 24.32 },
    { monthYear: "August 2022", cost: 22.11 },
    { monthYear: "September 2022", cost: 20.21 },
    { monthYear: "October 2022", cost: 21.14 },
    { monthYear: "November 2022", cost: 19.96 },
    { monthYear: "December 2022", cost: 20.52 }
];

const monthYearToNumber = (monthYear) => {
    const [month, year] = monthYear.split(' ');
    const monthIndex = new Date(Date.parse(month + " 1, 2022")).getMonth() + 1; // 1-based month index
    return (parseInt(year) - 2022) * 12 + monthIndex;
};

let x_vals = costData.map(d => monthYearToNumber(d.monthYear));
let y_vals = costData.map(d => d.cost);

let m, b;
const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

function setup() {
    createCanvas(500, 500);
    background(0);
    
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}

function predict(x_val) {
    return x_val.mul(m).add(b);
}

function project(x_val){
    return x_val = x_val*m+b
}

function draw() {
    tf.tidy(() => {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(tf.tensor1d(x_vals, 'float32')), ys));
    });
    background(0);
    stroke(255);
    strokeWeight(2);

    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], min(tfx), max(tfx), 0, width);
        let py = map(y_vals[i], 0, 1, height, 0);
        point(px, py);
    }

    const tfx = [0, 1];
    const tfy = tf.tidy(() => predict(tf.tensor1d(tfx, 'float32')));

    const lineY = tfy.dataSync();
    let x1 = map(tfx[0], min(tfx), max(tfx), 0, width);
    let x2 = map(tfx[1],  min(tfx), max(tfx), 0, width);
    let y1 = map(lineY[0],  min(tfy), max(tfy), height, 0);
    let y2 = map(lineY[1],  min(tfy), max(tfy),height, 0);

    line(x1, y1, x2, y2);
    strokeWeight(2);

    keyPressed()
}

function keyPressed() {
    let inputMonthYear = prompt("Enter month and year (e.g., 'January 2022'):");
    if (inputMonthYear) {
        let numericValue = monthYearToNumber(inputMonthYear);
        let prediction = tf.tidy(() => project(numericValue));

        background(0);
        stroke(255);
        fill(255);
        textSize(16);
        textAlign(CENTER, CENTER);
        text(`Prediction for ${inputMonthYear}: ${prediction}`, width / 2, height / 2);
    }
}
