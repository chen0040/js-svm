var jssvm = jssvm || {};

(function(jsr){
	var LinearSvm = function(config) {
        config = config || {};
        
        if(!config.iterations){
            config.iterations = 100;
        }
        if(!config.alpha){
            config.alpha = 0.0001;
        }
        if(!config.C){
            config.C = 1.0;
        }
        
        this.iterations = config.iterations;
        this.alpha = config.alpha;
        this.C = config.C;
    };
    
    LinearSvm.prototype.fit = function(data) {
        this.dim = data[0].length;
        var N = data.length;
        
        var X = [];
        var Y = [];
        for(var i=0; i < N; ++i){
            var row = data[i];
            var x_i = [];
            var y_i = row[row.length-1];
            x_i.push(1.0);
            for(var j=0; j < row.length-1; ++j){
                x_i.push(row[j]);
            }
            X.push(x_i);
            Y.push(y_i);
        }
        
        this.theta = [];
        for(var d = 0; d < this.dim; ++d){
            this.theta.push(0.0);
        }
        
        for(var iter = 0; iter < this.iterations; ++iter){
            var theta_delta = this.grad(X, Y, this.theta);
            for(var d = 0; d < this.dim; ++d){
                this.theta[d] = this.theta[d] - this.alpha * theta_delta[d];        
            }
        }
        
        return {
            theta: this.theta,
            cost: this.cost(X, Y, this.theta),
            config: {
                alpha: this.alpha,
                lambda: this.lambda,
                iterations: this.iterations 
            }
        }
    };
    
    LinearSvm.prototype.smallStep = function(theta, d) {
      return 0.00000000001;  
    };
    
    LinearSvm.prototype.grad = function(X, Y, theta) {
        var delta = [];
        var cost_now = this.cost(X, Y, theta);
        for(var d = 0; d < this.dim; ++d) {
            var small_step = this.smallStep(theta, d);
            var theta_d = theta[d];
            theta[d] += small_step;
            var cost_d = this.cost(X, Y, theta);
            theta[d] = theta_d;
            delta.push((cost_d - cost_now) / 2.0);
        }
        return delta;
    };
    
    LinearSvm.prototype.cost = function(X, Y, theta) {
        var N = X.length;
        var sum = 0;
        for(var i=0; i < N; ++i) {
            var x_i = X[i];
            var y_i = Y[i];
            sum += this.C * (y_i * this.cost1(x_i, y_i, theta) + (1-y_i) * this.cost0(x_i, y_i, theta));
        }
        
        sum += this.dotProduct(theta, theta) / 2.0;
    };
    
    LinearSvm.prototype.cost1 = function(x_i, y_i, theta) {
        var z = this.dotProduct(x_i, theta);
        if(z > 1){
            return 0;
        }
        return 1 - z;
    };
    
    LinearSvm.prototype.cost0 = function(x_i, y_i, theta) {
        var z = this.dotProduct(x_i, theta);
        if(z < -1) {
            return 0;
        }
        return 1 + z;
            
    };
    
    LinearSvm.prototype.dotProduct = function(x_i, theta) {
        var sum = 0;
        for(var d=0; d < this.dim; ++d){
            sum += x_i[j] * theta[j];
        }
        return sum;
    };
    
    LinearSvm.prototype.transform = function(x) {
        if(x[0].length){ // x is a matrix            
            var predicted_array = [];
            for(var i=0; i < x.length; ++i){
                var predicted = this.transform(x[i]);
                predicted_array.push(predicted);
            }
            return predicted_array;
        }
        
        var x_i = [];
        x_i.push(1.0);
        for(var j=0; j < x.length; ++j){
            x_i.push(x[j]);
        }
        
        return this.dotProduct(x_i, this.theta);  
    };
    
    
    jsr.LinearSvm = LinearSvm;

})(jssvm);

if(module) {
	module.exports = jssvm;
}