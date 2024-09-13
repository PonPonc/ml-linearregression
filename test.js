// Import TensorFlow.js (if using modules)
// import * as tf from '@tensorflow/tfjs';

// Example historical data (replace with your actual monthly data)
const materialCosts = [
    [10, 20, 30],  // Month 1
    [15, 25, 35],  // Month 2
    [20, 30, 40],  // Month 3
    [25, 35, 45],  // Month 4
    [30, 40, 50]   // Month 5
  ];
  
  const productionCosts = [100, 150, 200, 250, 300]; // Production costs corresponding to the material costs
  
  // Convert data to tensors
  const xs = tf.tensor2d(materialCosts, [materialCosts.length, 3]);
  const ys = tf.tensor2d(productionCosts, [productionCosts.length, 1]);
  
  // Define the model
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [3],
    units: 10,
    activation: 'relu'
  }));
  model.add(tf.layers.dense({
    units: 1
  }));
  
  // Compile the model
  model.compile({
    optimizer: 'sgd',
    loss: 'meanSquaredError'
  });
  
  // Train the model
  async function trainModel() {
    await model.fit(xs, ys, {
      epochs: 100
    });
    console.log('Model trained');
  }
  
  // Predict function
  async function predictNextMonth() {
    await trainModel();
    // Use the last month's material costs to predict the next month's production cost
    const lastMonthMaterialCosts = materialCosts[materialCosts.length - 1];
    const inputTensor = tf.tensor2d([lastMonthMaterialCosts], [1, 3]);
    const prediction = model.predict(inputTensor);
    prediction.print();
  }
  
  // Call predictNextMonth to get the prediction for the next month
  predictNextMonth();
  