import pandas as pd
import numpy as np

class TestHouses:

	def __init__(self, args):
		self.args = args
		df = pd.read_csv(args.test)
		self.mean = df.mean(axis=0, skipna=True)
		self.std = df.std(axis = 0)
		self.notes = [1]
		self.features = args.features
		self.questions = {  "Arithmancy": "How good are you at solving problems that require logical thinking ? Think math problems, enigmas and such... Rate yourself from 0 your brain is not wired that way to 10 you have a nobel prize in Mathematics",\
							"Astronomy": "Do you have a strong knowledge of celestial bodies ? Rate yourself from 0 you think the earth is flat to 10 you could explain at length and with great details everything we know about blackholes",\
							"Herbology": "Do you take drugs and how often ? Rate yourself from 0 never took a drug in your life to 10 you like to trip on LSD every two months or so. If you smoke weed and nothing else you're a 3 Karen.",\
							"Defense Against the Dark Arts": "If someone was being assaulted in front of you, would you react and how ? 2 points for calling the police 4 points for going up to the person and pretend you know them 2 points for rallying other people to go with you 2 points for physically defending the assaulted person. You may add points up if you would do several of the above",\
							"Divination": "How often do you say I told you so to your friends (and it's justified) ? 0 never 10 about every fucking day",\
							"Muggle Studies": "Do you often find yourself observing people in the subway ? 0 you only look at your phone 10 you're a total creep",\
							"Ancient Runes": "Do you often use secret code / languages to communicate with yourself or your friends ?",\
							"History of Magic": "How good is your knowledge of history ? 0 you're not sure what happened in 1989, 10 you often beat Akinator by using a random historical figure you know everything about",\
							"Transfiguration": "How good are you makeup skills ? 0 Inexistant 10 you could transform into another person",\
							"Potions": "How much/often do you drink ? 0 never 10 about three times a day",\
							"Care of Magical Creatures": "Are you an animal person ? 0 you hate them 10 you own a cat, a dog, a goat and three lovely tarentulas that you cherish deeply",\
							"Charms": "How charming are you ? 0 you're very unattractive 10 you're Keanu Reeves",\
							"Flying": "How high can you jump ? Every point = + 30 cm off the ground"\
							}
	
	def sigmoid(self, z):
		ret = 1 / (1 + np.exp(-z))
		return(ret)

	def get_valid_input(self, prompt):
		while True:
			try:
				value = int(input(prompt))
			except ValueError:
				print("Sorry, I need an integer.")
				continue
			if value < 0 or value > 10:
				print("Sorry, answer must be an integer between 0 and 10.")
				continue
			else:
				break
		return int(value)

	def magic_notes(self):
		for i in range(len(self.notes) - 1):
			self.notes[i + 1] = self.notes[i + 1] * (self.mean[self.features[i]] / 5.0)

	def normalize_notes(self):
		for i in range(len(self.notes) - 1):
			self.notes[i + 1] = (self.notes[i + 1] - (self.mean[self.features[i]])) / self.std[self.features[i]]

	def predict_house(self):
		print(self.notes)
		z = np.matmul(self.notes, self.args.thetas)
		print(z)
		H = self.sigmoid(z)
		print(H)
		print(self.args.houses)

	def launch_test(self):
		for ft in self.features:
			q = self.questions[ft]
			self.notes.append(self.get_valid_input(q))
		self.magic_notes()
		self.normalize_notes()
		self.predict_house()