const costData = [];

// Handle file input
document.getElementById('fileInput').addEventListener('change', handleFile, false);

function handleFile(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        const data = e.target.result;
        const workbook = XLSX.read(data, { type: 'binary' });

        // Array to hold all the formatted data
        costData.length = 0; 

        workbook.SheetNames.forEach(sheetName => {
            const sheet = workbook.Sheets[sheetName];
            const sheetData = XLSX.utils.sheet_to_json(sheet, { header: 1 });
            
            let monthYear = '';

            sheetData.forEach(row => {
                if (row.length === 0 || !row[0]) {
                    // Skip empty rows or rows with only one cell (potentially month headers)
                    return;
                }
                
                if (row[0] && typeof row[0] === 'string') {
                    monthYear = row[0];
                } else if (monthYear) {
                    costData.push({
                        itemcode: row[0],
                        description: row[1],
                        cost: row[2],
                        monthYear: monthYear
                    });
                }
            });
        });

        // Update graph data
        updateGraphData();

        // Log the formatted data
        console.log(JSON.stringify(costData, null, 2));
    };

    reader.readAsBinaryString(file);
}

// Convert monthYear string to a number
function monthYearToNumber(monthYearValue) {
    const [month, year] = monthYearValue.split(" ");
    const monthIndex = new Date(Date.parse(month + " 1, " + year)).getMonth() + 1;
    return (parseInt(year) - 2022) * 12 + monthIndex;
}

// Initialize and update x_vals and y_vals for graphing
let x_vals = [];
let y_vals = [];

function updateGraphData() {
    x_vals = costData
        .filter(data => data.itemcode === 4800088270588)
        .map(data => monthYearToNumber(data.monthYear));

    y_vals = costData
        .filter(data => data.itemcode === 4800088270588)
        .map(data => data.cost);
}

// TensorFlow.js setup
let m, b;

const learningRate = 0.2;
const optimizer = tf.train.adamax(learningRate);

function setup() {
    createCanvas(1000, 1000);
    background(0);

    // Initialize m and b as tensor variables with random weights
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}

function predict(x_val) {
    const xs = tf.tensor1d(x_val);
    return xs.mul(m).add(b);
}

//Predicts next month values based on current line of best fit
function projectedMonthCost(x_val) {
    const x = tf.scalar(x_val);
    const predictedValue = x.mul(m.dataSync()).add(b.dataSync());
    const value = predictedValue.dataSync();
    console.log(value)
    return tf.dispose(predictedValue);
     
}

function draw() {
    tf.tidy(() => {
        if (x_vals.length > 0) {
            const ys = tf.tensor1d(y_vals);

            // Perform optimization
            optimizer.minimize(() => {
                const pred = predict(x_vals);
                const lossValue = loss(pred, ys);
                return lossValue;
            });
        }
    });

    background(0);

    stroke(255);
    strokeWeight(10);
    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], 0, max(x_vals)+1, 0, width);
        let py = map(y_vals[i], 0, max(y_vals)+20, height, 0);
        point(px, py);
    }

    // Draw line of best fit
    const tfx = [Math.min(...x_vals), Math.max(...x_vals)];
    const tfy = tf.tidy(() => predict(tfx));
    const lineY = tfy.dataSync();

    tf.dispose(tfy);

    const x1 = map(tfx[0], 0, max(x_vals)+1, 0, width);
    const x2 = map(tfx[1], 0, max(x_vals)+1, 0, width);
    const y1 = map(lineY[0], 0, max(y_vals)+20, height, 0);
    const y2 = map(lineY[1], 0, max(y_vals)+20, height, 0);

    strokeWeight(2);
    line(x1, y1, x2, y2);
}
