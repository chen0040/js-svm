var jssvm = jssvm || {};

(function(jsr){
    jsr.small = function(){
        var one = 1.0;
        var two = 2.0;
        var tsmall = one / two;
        var z = tsmall * one + one;
        while(z > 1.0) {
            tsmall = tsmall / two;
            z = tsmall * one + one;
        }
        return tsmall * two * two;
    };
    
    jsr.quickSort = function(a, comparer){
        jsr._quickSort(a, 0, a.length-1, comparer);
    };
    
    jsr._quickSort = function(a, lo, hi, comparer) {
        if(lo >= hi) return;
        
        var j = jsr._partition(a, lo, hi, comparer);
        jsr._quickSort(a, lo, j-1, comparer);
        jsr._quickSort(a, j+1, hi, comparer);
    };
    
    jsr._partition = function(a, lo, hi, comparer){
        var v = a[lo];
        var i = lo, j = hi+1;
        while(true){
            while(comparer(v, a[++i]) > 0){
                if(i >= hi) break;
            }
            while(comparer(v, a[--j]) < 0) {
                if(j <= lo) break;
            }
            
            if(i >= j){
                break;
            }
            
            jsr._exchange(a, i, j);
        }
        
        jsr._exchange(a, lo, j);
        return j;
    };
    
    jsr._exchange = function(a, i, j) {
        var tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    };
    
    jsr.deltaStep = function(value){
        var rstep = Math.sqrt(this.small());  
        var temp = Math.max(1.0, Math.abs(value));
        var delta = Math.max(0.0, Math.abs(rstep * temp));
        if(value < 0) delta = -delta;
        return delta;
    };
    
    jsr.dotProduct = function(x_i, theta) {
        var sum = 0;
        var dim = theta.length;
        for(var d=0; d < dim; ++d){
            sum += x_i[d] * theta[d];
        }
        return sum;
    };
    
    jsr.gaussian = function(x_i, theta, L, sigma) {
        
        var dim = theta.length;
        
        var sum = 0;
        for(var d = 0; d < dim; ++d){
            var l_d = L[d];
            sum += theta[d] * jsr._gaussian(x_i, l_d, sigma, dim);
        }
        return sum;
    };
    
    jsr._gaussian = function(x_i, l_d, sigma, dim){
        var dx = [];
        for(var d = 0; d < dim; ++d){
            dx.push(x_i[d] - l_d[d]);
        }
        var z = jsr.dotProduct(dx, dx);
        return Math.exp(-z / (sigma * sigma));
    }
    
	var LinearSvm = function(config) {
        config = config || {};
        
        if(!config.iterations){
            config.iterations = 1000;
        }
        if(!config.alpha){
            config.alpha = 0.01;
        }
        if(!config.C){
            config.C = 5.0;
        }
        if(!config.trace){
            config.trace = false;
        }
        
        this.iterations = config.iterations;
        this.alpha = config.alpha;
        this.C = config.C;
        this.trace = config.trace;
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
            if(this.trace){
                console.log('cost: ' + this.cost(X, Y, this.theta));
            }
        }
        
        var scores = this.transform(data);
        
        this.threshold = null;
        
        var N = data.length;
        for(var i=0; i < N; ++i){
            var label = data[i][this.dim-1];
            if(label == 1){
                if(this.threshold == null || this.threshold > scores[i]){
                    this.threshold = scores[i];
                }
            }
        }
        
        return {
            theta: this.theta,
            cost: this.cost(X, Y, this.theta),
            threshold: this.threshold,
            config: {
                alpha: this.alpha,
                C: this.C,
                iterations: this.iterations,
                trace: this.trace
            }
        }
    };
    
    
    
    LinearSvm.prototype.smallStep = function(theta, d) {
      return jsr.deltaStep(theta[d]);  
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
        
        sum += jsr.dotProduct(theta, theta) / 2.0;
        
        sum /= N;
        return sum;
    };
    
    LinearSvm.prototype.cost1 = function(x_i, y_i, theta) {
        var z = jsr.dotProduct(x_i, theta);
        if(z > 1){
            return 0;
        }
        return 1 - z;
    };
    
    LinearSvm.prototype.cost0 = function(x_i, y_i, theta) {
        var z = jsr.dotProduct(x_i, theta);
        if(z < -1) {
            return 0;
        }
        return 1 + z;
            
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
        
        return jsr.dotProduct(x_i, this.theta);  
    };
    
    
    jsr.LinearSvm = LinearSvm;
    
    var KernelSvm = function(config) {
        config = config || {};
        
        if(!config.iterations){
            config.iterations = 1000;
        }
        if(!config.alpha){
            config.alpha = 0.01;
        }
        if(!config.C){
            config.C = 5.0;
        }
        if(!config.trace){
            config.trace = false;
        }
        if(!config.sigma) {
            config.sigma = 1.0;
        }
        
        this.iterations = config.iterations;
        this.alpha = config.alpha;
        this.C = config.C;
        this.trace = config.trace;
        this.sigma = config.sigma;
    };
    
    KernelSvm.prototype.fit = function(data) {
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
        this.L = [];
        for(var d = 0; d < this.dim; ++d){
            this.theta.push(0.0);
            this.L.push(X[d]);
        }
        
        for(var iter = 0; iter < this.iterations; ++iter){
            var theta_delta = this.grad(X, Y, this.theta);
            for(var d = 0; d < this.dim; ++d){
                this.theta[d] = this.theta[d] - this.alpha * theta_delta[d];        
            }
            if(this.trace){
                console.log('cost: ' + this.cost(X, Y, this.theta));
            }
        }
        
        var scores = this.transform(data);
        
        this.threshold = null;
        
        var N = data.length;
        for(var i=0; i < N; ++i){
            var label = data[i][this.dim-1];
            if(label == 1){
                if(this.threshold == null || this.threshold > scores[i]){
                    this.threshold = scores[i];
                }
            }
        }
        
        return {
            theta: this.theta,
            L: this.L,
            cost: this.cost(X, Y, this.theta),
            threshold: this.threshold,
            config: {
                alpha: this.alpha,
                C: this.C,
                iterations: this.iterations,
                trace: this.trace,
                sigma: this.sigma
            }
        }
    };
    
    
    
    KernelSvm.prototype.smallStep = function(theta, d) {
      return jsr.deltaStep(theta[d]);  
    };
    
    KernelSvm.prototype.grad = function(X, Y, theta) {

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
    
    KernelSvm.prototype.cost = function(X, Y, theta) {
        var N = X.length;
        var sum = 0;
        for(var i=0; i < N; ++i) {
            var x_i = X[i];
            var y_i = Y[i];
            sum += this.C * (y_i * this.cost1(x_i, y_i, theta) + (1-y_i) * this.cost0(x_i, y_i, theta));
        }
        
        sum += jsr.dotProduct(theta, theta) / 2.0;
        
        sum /= N;
        return sum;
    };
    
    KernelSvm.prototype.cost1 = function(x_i, y_i, theta) {
        var z = jsr.gaussian(x_i, theta, this.L, this.sigma);
        if(z > 1){
            return 0;
        }
        return 1 - z;
    };
    
    KernelSvm.prototype.cost0 = function(x_i, y_i, theta) {
        var z = jsr.gaussian(x_i, theta, this.L, this.sigma);
        if(z < -1) {
            return 0;
        }
        return 1 + z;
            
    };
    

    
    KernelSvm.prototype.transform = function(x) {
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
        for(var d = 1; d < this.dim; ++d){
            x_i.push(x[d-1]);
        }
        
        return jsr.gaussian(x_i, this.theta, this.L, this.sigma);  
    };
    
    
    jsr.KernelSvm = KernelSvm;
    
    var BinarySvmClassifier = function(config){
        var config = config || {};
        if(!config.kernel){
            config.kernel = 'linear';
        }
        if(!config.sigma) {
            config.sigma = 1.0;
        }
        this.kernel = config.kernel;
        if(config.kernel == 'linear'){
            this.classifier = new jsr.LinearSvm(config);
        } else if(config.kernel == 'gaussian') {
            this.classifier = new jsr.KernelSvm(config);
        } else {
            this.classifier = new jsr.LinearSvm(config);
        }
    };
    
    BinarySvmClassifier.prototype.fit = function(data){
        var result = this.classifier.fit(data);
        
        result.kernel = this.kernel;
        
        
        return result;
    }
    
    BinarySvmClassifier.prototype.transform = function(x) {
        return this.classifier.transform(x) > this.classifier.threshold ? 1 : 0;
    };
    
    jsr.BinarySvmClassifier = BinarySvmClassifier;
    
    var MultiClassSvmClassifier = function(config){
        var config = config || {};
        if(!config.alpha){
            config.alpha = 0.01;
        }
        if(!config.iterations) {
            config.iterations = 1000;
        }
        if(!config.C) {
            config.C = 5.0;
        }
        if(!config.kernel) {
            config.kernel = 'linear';
        }
        if(!config.sigma) {
            config.sigma = 1.0;
        }
        
        this.alpha = config.alpha;
        this.C = config.C;
        this.iterations = config.iterations;
        this.sigma = config.sigma;
        this.kernel = config.kernel;
    };
    
    MultiClassSvmClassifier.prototype.fit = function(data, classes) {
        this.dim = data[0].length;
        var N = data.length;
        
        if(!classes){
            classes = [];
            for(var i=0; i < N; ++i){
                var found = false;
                var label = data[i][this.dim-1];
                for(var j=0; j < classes.length; ++j){
                    if(label == classes[j]){
                        found = true;
                        break;
                    }
                }
                if(!found){
                    classes.push(label);
                }
            }
        }
        
        this.classes = classes;
        
        this.classifiers = {};
        var result = {};
        for(var k = 0; k < this.classes.length; ++k){
            var c = this.classes[k];
            
            if(this.kernel == 'linear'){
                this.classifiers[c] = new jsr.LinearSvm({
                    alpha: this.alpha,
                    C: this.C,
                    iterations: this.iterations
                });
            } else if(this.kernel == 'gaussian'){
                this.classifiers[c] = new jsr.KernelSvm({
                    alpha: this.alpha,
                    C: this.C,
                    iterations: this.iterations,
                    sigma: this.sigma
                });
            } else {
                this.classifiers[c] = new jsr.LinearSvm({
                    alpha: this.alpha,
                    C: this.C,
                    iterations: this.iterations
                });
            }
            var data_c = [];
            for(var i=0; i < N; ++i){
                var row = [];
                for(var j=0; j < this.dim-1; ++j){
                    row.push(data[i][j]);
                }
                row.push(data[i][this.dim-1] == c ? 1 : 0);
                data_c.push(row);
            }
            result[c] = this.classifiers[c].fit(data_c);
        }
        
        result.kernel = this.kernel;
        return result;
    };
    
    MultiClassSvmClassifier.prototype.transform = function(x) {
        if(x[0].length){ // x is a matrix            
            var predicted_array = [];
            for(var i=0; i < x.length; ++i){
                var predicted = this.transform(x[i]);
                predicted_array.push(predicted);
            }
            return predicted_array;
        }
        
        
        
        var max_prob = 0.0;
        var best_c = '';
        for(var k = 0; k < this.classes.length; ++k) {
            var c = this.classes[k];
            var prob_c;

            
            prob_c = this.classifiers[c].transform(x) - this.classifiers[c].threshold;
            if(max_prob <= prob_c){
                max_prob = prob_c;
                best_c = c;
            }
        }
        
        return best_c;
    }
    
    
    
    jsr.MultiClassSvmClassifier = MultiClassSvmClassifier;

})(jssvm);

var module = module || {};
if(module) {
	module.exports = jssvm;
}