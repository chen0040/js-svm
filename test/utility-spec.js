var expect    = require("chai").expect;
var jssvm = require("../src/jssvm");

describe("Test linear svm", function() {
  describe("Test utility functions", function() {
      
      it("small() will terminate", function() {
          var tsmall = jssvm.small();
          console.log('small: ' + tsmall);
      });
      
      it("deltaStep(100) should return a small positive number", function() {
          var delta = jssvm.deltaStep(100);
          console.log('delta: ' + delta);
          expect(delta).to.above(0);
      });
      
      it("deltaStep(0.1) should return a small positive number", function() {
          var delta = jssvm.deltaStep(0.1);
          console.log('delta: ' + delta);
          expect(delta).to.above(0);
      });
  });
    
    describe('quickSort function', function(){
       it('should sort descendingly', function(){
          a = [1, 4, 56, 2, 4, 2, 45, 3, 2, 534, 3, 7, 8];
           jssvm.quickSort(a, (a1, a2) => a1 - a2);
           for(var i=0; i < a.length; ++i){
               console.log(a[i]);
               if(i==0) continue;
               expect(a[i-1]).not.to.above(a[i]);
           }
       });
    });
});