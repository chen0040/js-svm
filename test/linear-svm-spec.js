var expect    = require("chai").expect;
var jssvm = require("../src/jssvm");

describe("Test linear svm", function() {
  describe("solve the coefficients in y = 2.0  + 5.0 * x", function() {
      var data = [];
      for(var x = 1.0; x < 100.0; x += 1.0) {
          var y = 2.0 + 5.0 * x + Math.random() * 1.0;
          data.push([x, y]);
      }
      
      var svm = new jssvm.LinearSvm({
          alpha: 0.001,
          iterations: 300,
          lambda: 0.0
      });
      var result = svm.fit(data);
      console.log(result);
      
      it("has a final cost of < 1.5", function(){
         expect(result.cost).to.below(1.5); 
      });
      
      it("its intercept should be about 2.0", function() {
          expect(svm.theta[0]).to.below(4.0); 
          expect(svm.theta[0]).to.above(0.0); 
      });
      
      it("its intercept should be about 5.0", function() {
          expect(svm.theta[1]).to.below(5.1); 
          expect(svm.theta[1]).to.above(4.9); 
      });
  });
});

