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
    { monthYear: "December 2022", cost: 20.52 },
    { monthYear: "January 2023", cost: 19.12 },
    { monthYear: "February 2023", cost: 21.51 },
    { monthYear: "March 2023", cost: 27.13 },
    { monthYear: "April 2023", cost: 18.17 },
    { monthYear: "May 2023", cost: 16.10 },
    { monthYear: "June 2023", cost: 17.45 },
    { monthYear: "July 2023", cost: 20.17 },
    { monthYear: "August 2023", cost: 19.18 },
    { monthYear: "September 2023", cost: 21.17 },
    { monthYear: "October 2023", cost: 21.00 },
    { monthYear: "November 2023", cost: 28.50 },
    { monthYear: "December 2023", cost: 23.12 }
];

//Month Year convertion
function monthYearToNumber(monthYearValue) {
    const [month, year] = monthYearValue.split(" ");
    const monthIndex = new Date(Date.parse(month + " " + year)).getMonth() + 1;
    return (parseInt(year) - 2022) * 12 + monthIndex;
}

//Initialized x and y values for graphing
let x_vals = costData.map(date => monthYearToNumber(date.monthYear));
let y_vals = costData.map(costs => costs.cost);

//Inialized m and b as slope and intercept values
let m, b;

//Initialized training fucnction and learningRate
const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);

//Setup Graph for visuals
function setup() {
    createCanvas(1000, 1000);
    background(0);

    //Initialized m and b as tensor variables with random wieghts
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));

}

//Calculates losses of all predictions and their corresponding y values
function loss(pred, labels) {
    //Values are squared to always have positive means
    return pred.sub(labels).square().mean();
}

//Predict function using y=mx+b formula to find line of best fit
function predict(x_val) {
    let xs = tf.tensor1d(x_val);
    return xs.mul(m).add(b);
}

//Predicts next month values based on current line of best fit
function projectedMonthCost(x_val) {
    const x = tf.scalar(x_val);
    const predictedValue = x.mul(m.dataSync()).add(b.dataSync());
    console.log(predictedValue.dataSync())
    return tf.dispose(predictedValue);
     
}

//Loop function to draw the graph and visualize the data
function draw() {
    tf.tidy(() => {
        //Starts the optimization process, if new data is uploaded then updates of best cost line will be done here.
        if (x_vals.length > 0) {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals), ys));
        }
    });

    background(0);

    //Maps the points of the data.
    stroke(255);
    strokeWeight(10);
    for (let i = 0; i < x_vals.length; i++) {
        //Added one values some fields to have the data be more centered on the display
        let px = map(x_vals[i], 0, max(x_vals) + 1, min(x_vals), width);
        let py = map(y_vals[i], 0, max(y_vals) + 20, height, min(y_vals));
        point(px, py);
    }

    //Initialize start and end points of the line, tfx is the starting and end points on the x axis
    const tfx = [1, max(x_vals)];
    //tfy is initialized as the y starting and end point of the y axis
    const tfy = tf.tidy(() => predict(tfx));
    let lineY = tfy.dataSync();
    tf.dispose(tfy);

    //Maping start and end points to the display
    let x1 = map(tfx[0], 0, max(x_vals) + 1, 0, width);
    let x2 = map(tfx[1], 0, max(x_vals) + 1, 0, width);
    let y1 = map(lineY[0], 0, max(y_vals) + 20, height, 0);
    let y2 = map(lineY[1], 0, max(y_vals) + 20, height, 0);

    strokeWeight(2);

    //Creates the line
    line(x1, y1, x2, y2);
}

