import json


class Mole:
    def __init__(self, asymmetric_score, size_score, border_score, color_score, final_score, classification_score, mole_center, mole_radius):
        self.asymmetric_score = asymmetric_score
        self.size_score = size_score
        self.border_score = border_score
        self.color_score = color_score
        self.final_score = final_score
        self.classification_score = classification_score
        self.mole_center = mole_center
        self.mole_radius = mole_radius
    
    def toJSON(self):
        return json.dumps(self, default=lambda o:o.__dict__, sort_keys=True, indent=4)
