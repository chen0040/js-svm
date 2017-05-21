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
});

