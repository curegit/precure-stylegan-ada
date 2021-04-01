
class AugmentationPipeline():

	def __init__(self, initial_probability=0.5):
		super().__init__()
		self.p = initial_probability
		self.manipulations = []

	def __call__(self, x):
		# pseudo
		for m in manipulations:
			for i in x:
				m(i) if random < p else i
