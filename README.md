# js-svm
Package provides javascript implementation of linear SVM and SVM with gaussian kernel

# Features

* Support for binary classification
* Support for multi-class classification 

# Install

```bash
npm install js-svm
```

# Usage

### SVM Binary Classifier

The sample code below show how to use SVM binary classifier on the iris datsets to classify whether a data row belong to species Iris-virginica:

```javascript
var jssvm = require('js-svm');
var iris = require('js-datasets-iris');

var svm = new jssvm.BinarySvmClassifier({
   alpha: 0.01,
   iterations: 1000,
   C: 5.0,
   trace: false
});

iris.shuffle();

var trainingDataSize = Math.round(iris.rowCount * 0.9);
var trainingData = [];
var testingData = [];
for(var i=0; i < iris.rowCount ; ++i) {
   var row = [];
   row.push(iris.data[i][0]); // sepalLength;
   row.push(iris.data[i][1]); // sepalWidth;
   row.push(iris.data[i][2]); // petalLength;
   row.push(iris.data[i][3]); // petalWidth;
   row.push(iris.data[i][4] == "Iris-virginica" ? 1.0 : 0.0); // output which is 1 if species is Iris-virginica; 0 otherwise
   if(i < trainingDataSize){
        trainingData.push(row);
   } else {
       testingData.push(row);
   }
}


var result = svm.fit(trainingData);

console.log(result);

for(var i=0; i < testingData.length; ++i){
   var predicted = svm.transform(testingData[i]);
   console.log("actual: " + testingData[i][4] + " predicted: " + predicted);
}
```


### Usage In HTML

Include the "node_modules/js-svm/build/jssvm.min.js" (or "node_modules/js-svm/src/jssvm.js") in your HTML \<script\> tag

The code in the script tag looks sth like this:

```javascript
var classifier = new jssvm.BinarySvmClassifier();
```

