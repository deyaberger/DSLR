import pandas as pd
import numpy as np
from termcolor import colored

class TestHouses:

	def __init__(self, args):
		self.args = args
		df = pd.read_csv(args.find)
		self.mean = df.mean(axis=0, skipna=True)
		self.std = df.std(axis = 0)
		self.notes = [1]
		self.features = args.features
		self.questions = {  "Arithmancy": "How good are you at solving problems that require logical thinking ? Think math problems, enigmas and such...\n0 : your brain is not wired that way\n10 : you have a nobel prize in Mathematics\n",\
							"Astronomy": "Do you have a strong knowledge of celestial bodies ?\n0 : you think the earth is flat\n10 : you could explain at length and with great details everything we know about blackholes\n",\
							"Herbology": "How often do you take drugs? 0 : No way I'm taking this shit - 10 : LSD every two months or so.\n",\
							"Defense Against the Dark Arts": "If someone was being assaulted in front of you, would you react and how (add up the points)?\n+2 points for calling the police\n+4 points for going up to the person and pretend you know them\n+2 points for rallying other people to go with you\n+2 points for physically defending the assaulted person.\n",\
							"Divination": "How often do you say 'I told you so' to your friends (and it's justified) ?\n0 : never\n10 : about every fucking day\n",\
							"Muggle Studies": "Do you often find yourself observing people in the subway ?\n0 : you only look at your phone\n10 : you're a total creep\n",\
							"Ancient Runes": "How often do you use secret code / languages to communicate with yourself or your friends ?\n",\
							"History of Magic": "How good is your knowledge of history ?\n0 : you're not sure what happened in 1989\n10 : you often beat Akinator by using a random historical figure you know everything about\n",\
							"Transfiguration": "How good are you makeup skills ?\n0 : Inexistant\n10 : you could transform into another person\n",\
							"Potions": "How much/often do you drink ?\n0: never\n10 : about three times a day\n",\
							"Care of Magical Creatures": "Are you an animal person ?\n0 : you hate them\n10 : you own a cat, a dog, a goat and three lovely tarentulas that you cherish deeply\n",\
							"Charms": "How charming are you ?\n0 : you're very unattractive\n10 : you're Keanu Reeves\n",\
							"Flying": "How high can you jump ?\nEvery point = + 20 cm off the ground\n"\
							}
	
	def sigmoid(self, z):
		ret = 1 / (1 + np.exp(-z))
		return(ret)

	def get_valid_input(self, prompt):
		while True:
			try:
				value = int(input(colored(prompt, 'cyan')))
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
		z = np.matmul(self.notes, self.args.thetas)
		H = self.sigmoid(z)
		return (H)

	def print_result(self, house):
		final_house = self.args.houses[np.argmax(house)]
		other_houses = []
		muggle = True
		for i in range(len(house)):
			if house[i] >= 0.9:
				muggle = False
				if self.args.houses[i] != final_house:
					other_houses.append(self.args.houses[i])
		if muggle == True:
			print(colored(f"You're a true muggle ! You have no affinity for magic whatsoever. The house you're closest to is {final_house} but honestly you're so far from getting in that you might as well stick to whatever else you're doing :)", 'red'))
		else :
			print(colored(f"Congrats ! You belong in {final_house}, your acceptance letter should come anytime soon... or not.", 'green'))
			if len(other_houses) > 0:
				print(colored(f"You were also pretty close from getting in {other_houses[0]}", 'blue'))
				for h in other_houses[1:]:
					print(colored(f"or {h}", 'blue'))
				print(colored(f"but {final_house} is very proud to count you amongst their members.", 'blue'))

	def print_notes(self):
		print(colored("\n\nHere's how you rated yourself :\n", 'magenta'))
		for i, note in enumerate(self.realnotes[1:]):
			print(colored(f"{self.features[i]} : {note}", 'white'))
		print()
		
	def launch_test(self):
		for ft in self.features:
			q = self.questions[ft]
			self.notes.append(self.get_valid_input(q))
		self.realnotes = self.notes.copy()
		self.magic_notes()
		self.normalize_notes()
		house = self.predict_house()
		self.print_notes()
		self.print_result(house)