//Product Cost data import.
const costData = [];
let currentMonthYear = '';

//Replace file name in fetch if location and name is different
// fetch('C:/Users/Razer/Downloads/v1-fg-costs.csv')
//   .then(response => response.text())
//   .then(csvText => {
document.getElementById('fileInput').addEventListener('change', function(e) {
  const csvFile = event.target.files[0]
  
  Papa.parse(csvFile, {
    complete: (parseData) => {
      parseData.data.forEach((row) => {
        if(row.length === 0 || (row.length === 1 && row[0] === '') || row[0] ==='Item Code' || row[0] === 'Description' || row[0] ==='Cost') return;

        if(row[0] && !row[2]){
          currentMonthYear = row[0]
        }

        else if (row[0] && row[2]){
          const productName = row[1]
          const productCost = parseFloat(row[2]);

          let monthYearEntry = costData.find(entry => entry.monthYear === currentMonthYear)
          if(!monthYearEntry){
            monthYearEntry = {monthYear: currentMonthYear, products:[]}
            costData.push(monthYearEntry)
          }

          monthYearEntry.products.push({
            productName: productName,
            cost: productCost
          })

          console.log(JSON.stringify(costData, null, 2))
          console.log(costData.filter(mY => mY.monthYear != '').map(d => d.monthYear))
          costData.forEach(entry => {
            entry.products.forEach(product => {
              console.log(`Product Name: ${product.productName}, Cost: ${product.cost}`);
            });
          });
        }
      })
    }
  })
})

//Month Year convertion to number
function monthYearToNumber(monthYearValue) {
  const [month, year] = monthYearValue.split(" ");
  const monthIndex = new Date(Date.parse(month + " 1, " + year)).getMonth() + 1;
  return (parseInt(year) - 2022) * 12 + monthIndex;
}

//Initializing model features
const monthYears = costData.map(d =>monthYearToNumber( d.monthYear));
const vch250gCost = costData.flatMap(entry => 
  entry.products.filter(product => product.productName === 'VIRGINIA Cocktail Hotdog 250g').map(product => product.cost)
);

const sh250gCost = costData.flatMap(entry => 
  entry.products.filter(product => product.productName === 'VIRGINIA Sweet Ham 250g').map(product => product.cost)
);

//Creating model
const model = tf.sequential()

model.add(tf.layers.dense({units: 64, activatoin: 'rele', inputShape: [numFeatures]}))
model.add(tf.layers.dense({units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 1}));

model.compile({
  optimizer: "adam",
  loss: 'meadnSquaredError'
})

