class Target():
    def __init__(self, model, loss_function, activate_function, threshold, first_n_byte = 1000000):
        self.model = model
        self.activate_function = activate_function
        self.loss_function = loss_function
        self.threshold = threshold
        self.first_n_byte = first_n_byte
    
    def predict(self, em_input):
        return self.model(em_input)

    def get_result(self, output, dim=None):
        if dim:
            result = (self.activate_function(output, dim=-1) > self.threshold).squeeze()
        else:
            result = (self.activate_function(output) > self.threshold).squeeze()
        return result