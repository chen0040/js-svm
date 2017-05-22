var expect    = require("chai").expect;
var jssvm = require("../src/jssvm");
var iris = require('js-datasets-iris');

describe("Test kernel svm", function() {

  describe("solve the binary classification problem of iris data for which species class == Iris-virginica", function(){
       var svm = new jssvm.KernelSvm({
           alpha: 0.01,
           iterations: 1000,
           C: 5.0,
           trace: false,
           sigma: 1.0
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
           var probabilityOfSpeciesBeingIrisVirginica = svm.transform(testingData[i]);
           console.log("kernel svm test: actual: " + testingData[i][4] + " probability of being Iris-virginica: " + probabilityOfSpeciesBeingIrisVirginica);
       }
        
       it('should have a cost of lower than 5', function(){
          expect(result.cost).to.below(5); 
       });
    });
});

   

