class Target():
    def __init__(self, model, threshold, first_n_byte = 1000000):
        self.model = model
        self.threshold = threshold
        self.first_n_byte = first_n_byte
    
    def predict(self, em_input):
        return self.model(em_input)

    def get_loss(self, outputs):
        return self.model.calculate_loss(outputs)

    def get_result(self, output):
        return self.model.get_result(output, self.threshold).squeeze()