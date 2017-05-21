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
        
    };
    
    
    jsr.LinearSvm = LinearSvm;

})(jssvm);

if(module) {
	module.exports = jssvm;
}