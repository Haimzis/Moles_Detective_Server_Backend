import json

class Mole:
    def __init__(self, asymetricResult, sizeResult, borderResult, coordinates):
        self.sizeResult = sizeResult
        self.borderResult = borderResult
        self.asymetricResult = asymetricResult
        self.coordinates = coordinates
    
    def toJSON(self):
        return json.dumps(self, default=lambda o:o.__dict__, sort_keys=True, indent=4)