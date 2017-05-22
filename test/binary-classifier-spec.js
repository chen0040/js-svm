var expect    = require("chai").expect;
var jssvm = require("../src/jssvm");
var iris = require('js-datasets-iris');

describe("Test binary svm classifier", function() {

  describe("Use linear svm to solve the binary classification problem of iris data for which species class == Iris-virginica", function(){
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
           console.log("linear svm binary classifier testing: actual: " + testingData[i][4] + " predicted: " + predicted);
       }
    });
    
    describe("Use kernel svm to solve the binary classification problem of iris data for which species class == Iris-virginica", function(){
       var svm = new jssvm.BinarySvmClassifier({
           alpha: 0.01,
           iterations: 1000,
           C: 5.0,
           trace: false,
           kernel: 'gaussian'
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
           console.log("kernel svm binary classifier testing: actual: " + testingData[i][4] + " predicted: " + predicted);
       }
    });
});

   