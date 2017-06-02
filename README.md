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

var svm = new jssvm.BinarySvmClassifier();

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

To configure the BinarySvmClassifier, use the following code when it is created:

```javascript
var svm = new jssvm.BinarySvmClassifier({
   alpha: 0.01, // learning rate
   iterations: 1000, // maximum iterations
   C: 5.0, // panelty term
   trace: false // debug tracing
});
```

### Multi-Class Classification using One-vs-All Logistic Regression

The sample code below illustrates how to run the multi-class classifier on the iris datasets to classifiy the species of each data row:

```javascript
var jssvm = require('js-svm');
var iris = require('js-datasets-iris');

var classifier = new jssvm.MultiClassSvmClassifier();

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
   row.push(iris.data[i][4]); // output is species
   if(i < trainingDataSize){
        trainingData.push(row);
   } else {
       testingData.push(row);
   }
}


var result = classifier.fit(trainingData);

console.log(result);

for(var i=0; i < testingData.length; ++i){
   var predicted = classifier.transform(testingData[i]);
   console.log("svm prediction testing: actual: " + testingData[i][4] + " predicted: " + predicted);
}
```

To configure the MultiClassSvmClassifier, use the following code when it is created:

```javascript
var classifier = new jssvm.MultiClassSvmClassifier({
   alpha: 0.01, // learning rate
   iterations: 1000, // maximum iterations
   C: 5.0 // panelty term
   sigma: 1.0 // the standard deviation for the gaussian kernel
});
```

### Switch between linear and guassian kernel

By default the kernel used by the binary and multi-class classifier is "linear" which can be printed by:

```javascript
console.log(classifier.kernel);
```

To switch to use gaussian kernel, put the property 'kernel: "gaussian"' in the config data when the classifier is created:

```javascript
var svm = new jssvm.BinarySvmClassifier({
   ...,
   kernel: 'gaussian'
});

....

var svm = new jssvm.MultiClassSvmClassifier({
   ...,
   kernel: 'gaussian'
});

```


### Usage In HTML

Include the "node_modules/js-svm/build/jssvm.min.js" (or "node_modules/js-svm/src/jssvm.js") in your HTML \<script\> tag

The demo code in HTML can be found in the following files within the package:

* [example-binary-classifier.html](https://rawgit.com/chen0040/js-svm/master/example-binary-classifier.html)
* [example-multi-class-classifier.html](https://rawgit.com/chen0040/js-svm/master/example-multi-class-classifier.html)


