var expect    = require("chai").expect;
var jssvm = require("../src/jssvm");
var iris = require('js-datasets-iris');

describe("Test multi-class classification using logistic regression", function(){
    describe("Use linear svm to solve the multi-class classification problem of iris data", function(){
       var classifier = new jssvm.MultiClassSvmClassifier({
           alpha: 0.001,
           iterations: 1000,
           C: 5.0
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
           console.log("linear svm multi-class prediction testing: actual: " + testingData[i][4] + " predicted: " + predicted);
       }
        
    });
    
    describe("Use kernel svm solve the multi-class classification problem of iris data", function(){
       var classifier = new jssvm.MultiClassSvmClassifier({
           alpha: 0.001,
           iterations: 1000,
           C: 5.0
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
           console.log("kernel svm multi-class prediction testing: actual: " + testingData[i][4] + " predicted: " + predicted);
       }
        
    });
});